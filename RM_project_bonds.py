
# ============================================================
# PROJECT - Valuation and Hedging – Bonds & Yields
# ============================================================

from io import StringIO
from datetime import datetime

import math
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0) USER PARAMETERS
# ============================================================
# The script automatically uses the latest available ECB date <= selected_date
selected_date = "2026-03-19"

# Bond universe
bond_category = "AAA rated bonds"

# Ten consecutive semi-annual maturities over 5 years
target_maturities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Semi-annual compounding
m = 2

# Face value of the STRIPS required by the assignment
par_value_strips = 100.0


# ============================================================
# 1) GENERIC HELPERS
# ============================================================
def maturity_from_code(code: str):
    """
    Convert an ECB code such as:
    - SR_6M
    - SR_1Y
    - SR_1Y6M
    - SR_18M
    - SR_2Y6M
    into a maturity in years.
    """
    if pd.isna(code):
        return None

    code = str(code).strip().upper()

    if not code.startswith("SR_"):
        return None

    raw = code[3:]

    if raw.endswith("M") and "Y" not in raw:
        months = int(raw[:-1])
        return months / 12.0

    if raw.endswith("Y") and "M" not in raw:
        years = int(raw[:-1])
        return float(years)

    match = re.fullmatch(r"(\d+)Y(\d+)M", raw)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months / 12.0

    return None


def almost_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def format_maturity_label(x):
    if float(x).is_integer():
        return f"{int(x)}Y"
    years = int(x)
    months = int(round((x - years) * 12))
    if years == 0:
        return f"{months}M"
    return f"{years}Y{months}M"


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# 2) ECB DATA DOWNLOAD
# ============================================================
def fetch_ecb_aaa_spot_curve(selected_date: str) -> pd.DataFrame:
    """
    Download AAA euro area government bond spot rates from the official ECB Data Portal.

    Dataset key:
    YC / B.U2.EUR.4F.G_N_A.SV_C_YM.
    where:
    - B       = Daily - business week
    - U2      = Euro area (changing composition)
    - EUR     = Euro
    - 4F      = ECB
    - G_N_A   = Government bond nominal AAA
    - SV_C_YM = Svensson model - continuous compounding - yield error minimisation
    """
    base_url = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM."

    end_dt = pd.to_datetime(selected_date)
    start_dt = end_dt - pd.Timedelta(days=31)

    params = {
        "startPeriod": start_dt.strftime("%Y-%m-%d"),
        "endPeriod": end_dt.strftime("%Y-%m-%d"),
        "format": "csvdata"
    }

    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = [c.strip().upper() for c in df.columns]

    possible_date_cols = ["TIME_PERIOD", "TIME PERIOD", "DATE", "OBS_DATE"]
    possible_value_cols = ["OBS_VALUE", "OBS VALUE", "VALUE"]

    date_col = next((c for c in possible_date_cols if c in df.columns), None)
    value_col = next((c for c in possible_value_cols if c in df.columns), None)

    if date_col is None or value_col is None:
        raise ValueError(
            f"Could not identify date/value columns. Columns found: {list(df.columns)}"
        )

    key_col = "KEY" if "KEY" in df.columns else None

    fm_type_col = None
    if key_col is None:
        for c in df.columns:
            sample = df[c].astype(str).head(50).tolist()
            if any("SR_" in s for s in sample):
                fm_type_col = c
                break

    if key_col is not None:
        df["SERIES_CODE"] = df[key_col].astype(str).str.split(".").str[-1]
    elif fm_type_col is not None:
        df["SERIES_CODE"] = df[fm_type_col].astype(str)
    else:
        raise ValueError(
            "Could not identify the ECB series code (SR_*). "
            f"Available columns: {list(df.columns)}"
        )

    df = df[df["SERIES_CODE"].astype(str).str.startswith("SR_")].copy()
    df["MATURITY"] = df["SERIES_CODE"].apply(maturity_from_code)
    df = df[df["MATURITY"].notna()].copy()

    df["DATE"] = pd.to_datetime(df[date_col])
    df["RATE"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["RATE"].notna()].copy()

    df = df[df["MATURITY"].isin(target_maturities)].copy()

    if df.empty:
        raise ValueError("No AAA spot-rate observations were retrieved for the requested maturities.")

    selected_dt = pd.to_datetime(selected_date)
    df = df[df["DATE"] <= selected_dt].copy()

    if df.empty:
        raise ValueError("No ECB observations are available at a date <= selected_date.")

    df = df.sort_values(["MATURITY", "DATE"])
    latest = df.groupby("MATURITY", as_index=False).tail(1).copy()
    latest = latest.sort_values("MATURITY").reset_index(drop=True)

    missing = [x for x in target_maturities if x not in latest["MATURITY"].tolist()]
    if missing:
        raise ValueError(
            "Some maturities could not be retrieved from the ECB: "
            + ", ".join(format_maturity_label(float(x)) for x in missing)
        )

    return latest[["DATE", "MATURITY", "SERIES_CODE", "RATE"]].copy()


# ============================================================
# 3) BOND-SPECIFIC HELPERS FOR QUESTIONS 7 TO 11
# ============================================================
def get_spot_rate_for_maturity(t, maturity_grid, spot_rate_grid):
    for mt, sr in zip(maturity_grid, spot_rate_grid):
        if almost_equal(mt, t):
            return sr
    raise ValueError(f"No spot rate found for maturity {t}Y.")


def get_forward_rate_for_maturity(t, maturity_grid, forward_rate_grid):
    for mt, fr in zip(maturity_grid, forward_rate_grid):
        if almost_equal(mt, t):
            return fr
    raise ValueError(f"No forward rate found for maturity {t}Y.")


def bond_coupon(face_value, coupon_rate, frequency):
    return face_value * coupon_rate / frequency


def bond_price_spot(face_value, coupon_rate, maturity_years, frequency, maturity_grid, spot_rate_grid):
    """
    Price a coupon bond with spot rates:
    P = sum( CF_t / (1 + s_t/m)^(m*t) )
    """
    coupon = bond_coupon(face_value, coupon_rate, frequency)
    n_periods = int(round(maturity_years * frequency))
    price = 0.0

    for k in range(1, n_periods + 1):
        t = k / frequency
        s_t = get_spot_rate_for_maturity(t, maturity_grid, spot_rate_grid)

        if k < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value

        price += cf / ((1 + s_t / frequency) ** (frequency * t))

    return price


def bond_price_forward(face_value, coupon_rate, maturity_years, frequency, maturity_grid, forward_rate_grid):
    """
    Price a coupon bond with forward rates:
    DF_t = product_{j=1..k} 1 / (1 + f_j/m)
    """
    coupon = bond_coupon(face_value, coupon_rate, frequency)
    n_periods = int(round(maturity_years * frequency))
    price = 0.0
    discount_factor = 1.0

    for k in range(1, n_periods + 1):
        t = k / frequency
        f_t = get_forward_rate_for_maturity(t, maturity_grid, forward_rate_grid)
        discount_factor = discount_factor / (1 + f_t / frequency)

        if k < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value

        price += cf * discount_factor

    return price


def bond_dirty_price_from_settlement(face_value, coupon_rate, maturity_years, frequency,
                                     maturity_grid, spot_rate_grid,
                                     days_since_last_coupon, coupon_period_days=181):
    """
    Dirty price between coupon dates using spot rates.

    Assumptions:
    - The issue date is also the last coupon date.
    - Settlement occurs before the first coupon date.
    - The first coupon is in 0.5 year.
    """
    coupon = bond_coupon(face_value, coupon_rate, frequency)
    n_periods = int(round(maturity_years * frequency))

    alpha = days_since_last_coupon / coupon_period_days
    w = 1 - alpha

    dirty_price = 0.0

    for k in range(1, n_periods + 1):
        t = k / frequency
        s_t = get_spot_rate_for_maturity(t, maturity_grid, spot_rate_grid)

        if k < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value

        exponent = (k - 1) + w
        dirty_price += cf / ((1 + s_t / frequency) ** exponent)

    accrued_interest = coupon * alpha
    clean_price = dirty_price - accrued_interest

    return dirty_price, clean_price, accrued_interest, alpha, w


def bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm):
    """
    Price a coupon bond with a single yield-to-maturity:
    P = sum( CF_k / (1 + y/m)^k )
    """
    coupon = bond_coupon(face_value, coupon_rate, frequency)
    n_periods = int(round(maturity_years * frequency))
    price = 0.0

    for k in range(1, n_periods + 1):
        if k < n_periods:
            cf = coupon
        else:
            cf = coupon + face_value

        price += cf / ((1 + ytm / frequency) ** k)

    return price


def solve_ytm_bisection(target_price, face_value, coupon_rate, maturity_years, frequency,
                        low=-0.50, high=0.50, tol=1e-12, max_iter=1000):
    """
    Solve the YTM with a bisection method without financial libraries.
    """
    def f(y):
        return bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, y) - target_price

    f_low = f(low)
    f_high = f(high)

    expand_count = 0
    while f_low * f_high > 0 and expand_count < 50:
        low -= 0.25
        high += 0.25
        f_low = f(low)
        f_high = f(high)
        expand_count += 1

    if f_low * f_high > 0:
        raise ValueError("The bisection interval does not bracket the YTM root.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = f(mid)

        if abs(f_mid) < tol:
            return mid

        if f_low * f_mid < 0:
            high = mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def dv01_from_ytm(face_value, coupon_rate, maturity_years, frequency, ytm, shift_bp=5):
    """
    DV01 for a yield decrease of shift_bp.
    DV01 = - DeltaBV / (10000 * DeltaY)
    """
    p0 = bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm)
    dy = -shift_bp / 10000
    p_shift = bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm + dy)
    delta_bv = p_shift - p0
    dv01 = - delta_bv / (10000 * dy)
    return p0, p_shift, delta_bv, dv01


def duration_convexity_from_ytm(face_value, coupon_rate, maturity_years, frequency, ytm, shift=0.0055):
    """
    Effective duration and convexity using symmetric yield shocks.
    """
    p0 = bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm)
    p_down = bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm - shift)
    p_up = bond_price_ytm(face_value, coupon_rate, maturity_years, frequency, ytm + shift)

    duration = (p_down - p_up) / (2 * p0 * shift)
    convexity = (p_down + p_up - 2 * p0) / (p0 * shift ** 2)

    return p0, p_down, p_up, duration, convexity


def price_change_duration_convexity(price, duration, convexity, dy):
    """
    DeltaP/P ≈ -D*dy + 0.5*C*dy^2
    """
    pct_change = -duration * dy + 0.5 * convexity * dy ** 2
    amount_change = price * pct_change
    new_price = price + amount_change
    return pct_change, amount_change, new_price


def bond_risk_profile(face_value, coupon_rate, maturity_years, frequency, maturity_grid, spot_rate_grid):
    """
    Compute price, YTM, DV01, duration, and convexity for a hedge bond.
    """
    price = bond_price_spot(face_value, coupon_rate, maturity_years, frequency, maturity_grid, spot_rate_grid)
    ytm = solve_ytm_bisection(price, face_value, coupon_rate, maturity_years, frequency)

    _, _, _, dv01 = dv01_from_ytm(face_value, coupon_rate, maturity_years, frequency, ytm, shift_bp=5)
    _, _, _, duration, convexity = duration_convexity_from_ytm(
        face_value, coupon_rate, maturity_years, frequency, ytm, shift=0.0055
    )

    return {
        "price": price,
        "ytm": ytm,
        "dv01": dv01,
        "duration": duration,
        "convexity": convexity
    }


# ============================================================
# 4) GET DATA FROM THE ECB
# ============================================================
ecb_data = fetch_ecb_aaa_spot_curve(selected_date)

actual_date_used = ecb_data["DATE"].iloc[0].strftime("%Y-%m-%d")
maturities = ecb_data["MATURITY"].tolist()
spot_rates_cc_pct = ecb_data["RATE"].tolist()
spot_rates_cc = [x / 100 for x in spot_rates_cc_pct]

# The ECB dataset uses continuous compounding (SV_C_YM).
# Convert to semi-annual compounding for use in all formulas:
# (1 + s_sa / m)^m = e^{s_cc}  =>  s_sa = m * (exp(s_cc / m) - 1)
spot_rates = [m * (math.exp(r / m) - 1) for r in spot_rates_cc]
spot_rates_pct = [r * 100 for r in spot_rates]

print_section("ECB DATA EXTRACTED")
print(f"Target date requested : {selected_date}")
print(f"Actual ECB date used  : {actual_date_used}")
print(f"Bond category         : {bond_category}")
print(f"\nNote: ECB rates use continuous compounding (SV_C_YM).")
print("They have been converted to semi-annual compounding for all calculations.\n")
print("Extracted and converted spot rates:")
for mat, code, rate_cc, rate_sa in zip(ecb_data["MATURITY"], ecb_data["SERIES_CODE"], spot_rates_cc_pct, spot_rates_pct):
    print(f"Maturity {mat:>3.1f} years | Series {code:<8} | CC yield = {rate_cc:.6f}% | Semi-annual yield = {rate_sa:.6f}%")

ecb_data.to_csv("ecb_aaa_spot_curve_selected_date.csv", index=False)


# ============================================================
# QUESTION 1
# ============================================================
q1_text = """
QUESTION 1 - DEFINE A BOND AND EXPLAIN HOW IT IS ISSUED.
EXPLAIN THE RELATIONSHIP BETWEEN BONDS AND INTEREST RATES.

A bond is a fixed-income security representing a loan made by an investor
to an issuer such as a government, a municipality, or a corporation.
The issuer borrows money by issuing bonds on the market and promises:
1) to pay periodic coupons,
2) to repay the principal (par value / face value) at maturity.

Bonds are generally issued at par value. Once issued, their market price
changes according to economic conditions, especially changes in interest rates.

Relationship between bond prices and interest rates:
- when interest rates rise, existing bond prices fall;
- when interest rates fall, existing bond prices rise.

This inverse relationship exists because investors compare existing bonds
with newly issued bonds offering current market yields.
"""

print_section("QUESTION 1")
print(q1_text)


# ============================================================
# QUESTION 2
# ============================================================
print_section("QUESTION 2 - SELECTED DATE AND 10 CONSECUTIVE SEMI-ANNUAL SPOT YIELDS")
print(f"Chosen bond category : {bond_category}")
print(f"Requested date       : {selected_date}")
print(f"ECB date used        : {actual_date_used}")

q2_table = pd.DataFrame({
    "date_used": [actual_date_used] * len(maturities),
    "maturity_years": maturities,
    "spot_rate_pct": spot_rates_pct
})

print("\nSelected spot yields:")
print(q2_table.to_string(index=False))
q2_table.to_csv("question_2_spot_yields.csv", index=False)


# ============================================================
# QUESTION 3
# ============================================================
q3_text = """
QUESTION 3 - DEFINE A SPOT RATE AND ITS CHARACTERISTICS.
EXPLAIN HOW SPOT RATES ARE DERIVED FROM THE MARKET.
PLOT THE SPOT YIELD CURVE. INTERPRET IT FROM A MACROECONOMIC PERSPECTIVE.

Definition:
A spot rate is the interest rate applicable today for a single cash flow
received at a specific maturity. It is also called a zero-coupon rate.

Main characteristics:
- it discounts one single future cash flow to today;
- it is specific to one maturity;
- the set of spot rates across maturities forms the spot yield curve.

Derivation from the market:
Spot rates are derived by bootstrapping from zero-coupon bonds (STRIPS)
or from coupon-bearing instruments. For a zero-coupon bond:
    y = m * ((M / P) ** (1 / (m*n)) - 1)
where:
    y = spot rate
    M = par value
    P = zero-coupon bond price
    n = maturity in years
    m = number of periods per year
"""

macro_comment = """
The shape of the spot yield curve can be interpreted from a macroeconomic perspective.
If the curve is upward sloping, long-term yields are above short-term yields, which usually
reflects expectations of continued economic activity, inflation compensation, and a term premium.
If the curve is downward sloping, markets may be pricing weaker future growth, lower inflation,
or future monetary easing. A relatively flat curve suggests limited differences between short-term
and long-term rate expectations over the selected horizon.
"""

print_section("QUESTION 3")
print(q3_text)
print("Macroeconomic interpretation:")
print(macro_comment)

plt.figure(figsize=(9, 5))
plt.plot(maturities, spot_rates_pct, marker="o")
plt.title(f"ECB AAA Spot Yield Curve - {actual_date_used}")
plt.xlabel("Maturity (years)")
plt.ylabel("Spot yield (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("question_3_spot_curve.png", dpi=300)
plt.show()


# ============================================================
# QUESTION 4
# ============================================================
x1 = 1.5
x2 = 2.5
xi = 2.0

y1 = spot_rates_pct[maturities.index(x1)]
y2 = spot_rates_pct[maturities.index(x2)]
observed_yi = spot_rates_pct[maturities.index(xi)]

estimated_yi = y1 + (y2 - y1) * ((xi - x1) / (x2 - x1))
interpolation_error_pct = estimated_yi - observed_yi
interpolation_error_bp = interpolation_error_pct * 100

print_section("QUESTION 4 - LINEAR INTERPOLATION")
print(f"Known spot yield 1: maturity {x1:.1f}Y -> {y1:.6f}%")
print(f"Known spot yield 2: maturity {x2:.1f}Y -> {y2:.6f}%")
print(f"Estimated maturity : {xi:.1f}Y")

print("\nLinear interpolation formula:")
print("yi = y1 + (y2 - y1) * ((xi - x1) / (x2 - x1))")

print(f"\nEstimated spot yield at {xi:.1f}Y : {estimated_yi:.6f}%")
print(f"Observed spot yield at {xi:.1f}Y  : {observed_yi:.6f}%")
print(f"Difference                        : {interpolation_error_pct:.6f}% ({interpolation_error_bp:.2f} bps)")

if abs(interpolation_error_bp) < 5:
    interp_comment = (
        "The interpolated value is very close to the observed value, which suggests that the spot curve "
        "is locally smooth between the two chosen maturities."
    )
else:
    interp_comment = (
        "The gap between the interpolated value and the observed value is more significant, which shows "
        "that the local shape of the curve is not perfectly linear."
    )

print("\nInterpretation:")
print(interp_comment)


# ============================================================
# QUESTION 5
# ============================================================
strips_prices = []
for n, y in zip(maturities, spot_rates):
    price = par_value_strips / ((1 + y / m) ** (m * n))
    strips_prices.append(price)

q5_table = pd.DataFrame({
    "maturity_years": maturities,
    "spot_rate_pct": spot_rates_pct,
    "strips_price_par_100": strips_prices,
    "position_vs_par": ["Below par" if p < par_value_strips else "Above par" for p in strips_prices]
})

print_section("QUESTION 5 - PRICES OF THE 10 STRIPS IMPLIED BY THE SPOT RATES")
print(q5_table.to_string(index=False))
q5_table.to_csv("question_5_strips_prices.csv", index=False)

print("\nInterpretation:")
print(
    "If the spot yield is positive, the corresponding zero-coupon bond is priced below par. "
    "This is because a STRIP pays no coupons, so its return comes entirely from the difference "
    "between its discounted purchase price and its par value received at maturity."
)


# ============================================================
# QUESTION 6
# ============================================================
forward_rates = []
for i, (t, s_t) in enumerate(zip(maturities, spot_rates)):
    if i == 0:
        f_t = s_t
    else:
        t_prev = maturities[i - 1]
        s_prev = spot_rates[i - 1]

        l_t = m * t
        l_prev = m * t_prev

        growth_t = (1 + s_t / m) ** l_t
        growth_prev = (1 + s_prev / m) ** l_prev
        f_t = m * ((growth_t / growth_prev) - 1)

    forward_rates.append(f_t)

forward_rates_pct = [f * 100 for f in forward_rates]

q6_text = """
QUESTION 6 - DEFINE A FORWARD RATE AND ITS CHARACTERISTICS.
EXPLAIN HOW FORWARD RATES ARE DERIVED FROM THE MARKET.

Definition:
A forward rate is the interest rate agreed today for a loan or investment
that will start at a future date and last for one period.

Main characteristics:
- it concerns a future period rather than today;
- it is derived from spot rates;
- it is useful to infer market expectations embedded in the yield curve.

Derivation from the market:
Forward rates are derived from the no-arbitrage relationship between spot rates
at consecutive maturities.
"""

q6_table = pd.DataFrame({
    "maturity_years": maturities,
    "spot_rate_pct": spot_rates_pct,
    "forward_rate_pct": forward_rates_pct
})

print_section("QUESTION 6")
print(q6_text)
print(q6_table.to_string(index=False))
q6_table.to_csv("question_6_forward_rates.csv", index=False)

forward_slope = forward_rates_pct[-1] - forward_rates_pct[0]
if forward_slope > 0.15:
    forward_comment = (
        "The forward curve is upward sloping. This suggests that the market prices in higher future "
        "short-period rates over the selected horizon."
    )
elif forward_slope < -0.15:
    forward_comment = (
        "The forward curve is downward sloping. This suggests that the market prices in lower future "
        "short-period rates over the selected horizon."
    )
else:
    forward_comment = (
        "The forward curve is relatively flat over the selected horizon. This suggests limited variation "
        "in expected future short-period rates."
    )

print("\nInterpretation:")
print(forward_comment)

plt.figure(figsize=(9, 5))
plt.plot(maturities, forward_rates_pct, marker="o")
plt.title(f"ECB AAA Forward Curve - {actual_date_used}")
plt.xlabel("Maturity (years)")
plt.ylabel("Forward rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("question_6_forward_curve.png", dpi=300)
plt.show()


# ============================================================
# QUESTION 7
# ============================================================
print_section("QUESTION 7 - PRICE OF THE 3-YEAR FRENCH GOVERNMENT BOND")

face_value_q7 = 1000.0
coupon_rate_q7 = 0.02
maturity_q7 = 3.0
frequency_q7 = 2

price_q7_spot = bond_price_spot(
    face_value_q7, coupon_rate_q7, maturity_q7, frequency_q7,
    maturities, spot_rates
)

price_q7_forward = bond_price_forward(
    face_value_q7, coupon_rate_q7, maturity_q7, frequency_q7,
    maturities, forward_rates
)

q7_table = pd.DataFrame([
    {"sub_question": "7a", "metric": "Bond price using spot rates", "value": price_q7_spot},
    {"sub_question": "7b", "metric": "Bond price using forward rates", "value": price_q7_forward},
    {"sub_question": "7c", "metric": "Difference (spot - forward)", "value": price_q7_spot - price_q7_forward},
])

print(q7_table.to_string(index=False))
q7_table.to_csv("question_7_bond_prices.csv", index=False)

print("\nInterpretation:")
print(
    "The two prices should be equal, up to very small numerical rounding differences, because spot rates "
    "and forward rates are linked by no-arbitrage relationships."
)


# ============================================================
# QUESTION 8
# ============================================================
print_section("QUESTION 8 - DIRTY PRICE AND CLEAN PRICE AFTER 50 DAYS")

days_since_issue = 50
coupon_period_days_q8 = (datetime(2025, 7, 1) - datetime(2025, 1, 1)).days

dirty_q8, clean_q8, accrued_q8, alpha_q8, w_q8 = bond_dirty_price_from_settlement(
    face_value_q7, coupon_rate_q7, maturity_q7, frequency_q7,
    maturities, spot_rates,
    days_since_last_coupon=days_since_issue,
    coupon_period_days=coupon_period_days_q8
)

q8_table = pd.DataFrame([
    {"metric": "Days since issue / last coupon", "value": days_since_issue},
    {"metric": "Coupon period days", "value": coupon_period_days_q8},
    {"metric": "Accrual fraction alpha", "value": alpha_q8},
    {"metric": "Remaining fraction w", "value": w_q8},
    {"metric": "Accrued interest (EUR)", "value": accrued_q8},
    {"metric": "Dirty price (EUR)", "value": dirty_q8},
    {"metric": "Clean price (EUR)", "value": clean_q8},
])

print(q8_table.to_string(index=False))
q8_table.to_csv("question_8_dirty_clean_price.csv", index=False)

print("\nInterpretation:")
print(
    "The dirty price includes accrued interest. The clean price excludes accrued interest and is therefore "
    "lower than the dirty price."
)


# ============================================================
# QUESTION 9
# ============================================================
print_section("QUESTION 9 - YIELD TO MATURITY")

ytm_q9 = solve_ytm_bisection(
    price_q7_spot,
    face_value_q7,
    coupon_rate_q7,
    maturity_q7,
    frequency_q7
)

spot_3y = get_spot_rate_for_maturity(3.0, maturities, spot_rates)

q9_table = pd.DataFrame([
    {"metric": "Bond price from question 7a (EUR)", "value": price_q7_spot},
    {"metric": "Estimated YTM", "value": ytm_q9},
    {"metric": "3-year spot rate", "value": spot_3y},
])

print(q9_table.to_string(index=False))
q9_table.to_csv("question_9_ytm.csv", index=False)

print("\nInterpretation:")
print(
    "The YTM is the single discount rate that equates the present value of all future cash flows to the "
    "observed bond price. It is a summary yield, whereas the spot curve uses one rate for each maturity."
)


# ============================================================
# QUESTION 10
# ============================================================
print_section("QUESTION 10 - DV01, DURATION, AND CONVEXITY")

p0_q10a, p_shift_q10a, delta_bv_q10a, dv01_q10a = dv01_from_ytm(
    face_value_q7, coupon_rate_q7, maturity_q7, frequency_q7, ytm_q9, shift_bp=5
)

p0_q10b, p_down_q10b, p_up_q10b, duration_q10b, convexity_q10b = duration_convexity_from_ytm(
    face_value_q7, coupon_rate_q7, maturity_q7, frequency_q7, ytm_q9, shift=0.0055
)

dy_large = 0.018

pct_down_18, amount_down_18, new_price_down_18 = price_change_duration_convexity(
    p0_q10b, duration_q10b, convexity_q10b, -dy_large
)
pct_up_18, amount_up_18, new_price_up_18 = price_change_duration_convexity(
    p0_q10b, duration_q10b, convexity_q10b, dy_large
)

q10a_table = pd.DataFrame([
    {"metric": "Initial price (EUR)", "value": p0_q10a},
    {"metric": "Shifted price after -5 bps (EUR)", "value": p_shift_q10a},
    {"metric": "Price change (EUR)", "value": delta_bv_q10a},
    {"metric": "DV01 (EUR per 1 bp)", "value": dv01_q10a},
])

q10b_table = pd.DataFrame([
    {"metric": "Price if YTM decreases by 0.55% (EUR)", "value": p_down_q10b},
    {"metric": "Price if YTM increases by 0.55% (EUR)", "value": p_up_q10b},
    {"metric": "Duration", "value": duration_q10b},
    {"metric": "Convexity", "value": convexity_q10b},
])

q10c_table = pd.DataFrame([
    {"scenario": "YTM decreases by 1.8%", "pct_change": pct_down_18, "amount_change_eur": amount_down_18, "estimated_price_eur": new_price_down_18},
    {"scenario": "YTM increases by 1.8%", "pct_change": pct_up_18, "amount_change_eur": amount_up_18, "estimated_price_eur": new_price_up_18},
])

print("10a) DV01 for a 5 bps decrease in YTM")
print(q10a_table.to_string(index=False))

print("\n10b) Duration and convexity for a +/-0.55% change in YTM")
print(q10b_table.to_string(index=False))

print("\n10c) Price change for a +/-1.8% change in YTM using duration/convexity")
print(q10c_table.to_string(index=False))

q10a_table.to_csv("question_10a_dv01.csv", index=False)
q10b_table.to_csv("question_10b_duration_convexity.csv", index=False)
q10c_table.to_csv("question_10c_large_shock.csv", index=False)

print("\nInterpretation:")
print(
    "DV01 measures the first-order euro sensitivity of the bond to a 1 basis point yield change. "
    "Duration captures first-order proportional sensitivity, while convexity captures the curvature "
    "of the price-yield relationship and improves the approximation for larger shocks."
)


# ============================================================
# QUESTION 11
# ============================================================
print_section("QUESTION 11 - HEDGING STRATEGIES")

open_position_value = 7_300_000.0

# Illustrative hedge bonds
hedge_1_face = 1000.0
hedge_1_coupon = 0.015
hedge_1_maturity = 2.0
hedge_1_freq = 2

hedge_2_face = 1000.0
hedge_2_coupon = 0.03
hedge_2_maturity = 5.0
hedge_2_freq = 2

open_profile = {
    "price": price_q7_spot,
    "ytm": ytm_q9,
    "dv01": dv01_q10a,
    "duration": duration_q10b,
    "convexity": convexity_q10b
}

hedge_1_profile = bond_risk_profile(
    hedge_1_face, hedge_1_coupon, hedge_1_maturity, hedge_1_freq, maturities, spot_rates
)

hedge_2_profile = bond_risk_profile(
    hedge_2_face, hedge_2_coupon, hedge_2_maturity, hedge_2_freq, maturities, spot_rates
)

q11_instruments_table = pd.DataFrame([
    {
        "instrument": "Open position bond",
        "price_eur": open_profile["price"],
        "ytm": open_profile["ytm"],
        "dv01": open_profile["dv01"],
        "duration": open_profile["duration"],
        "convexity": open_profile["convexity"]
    },
    {
        "instrument": "Hedge bond 1",
        "price_eur": hedge_1_profile["price"],
        "ytm": hedge_1_profile["ytm"],
        "dv01": hedge_1_profile["dv01"],
        "duration": hedge_1_profile["duration"],
        "convexity": hedge_1_profile["convexity"]
    },
    {
        "instrument": "Hedge bond 2",
        "price_eur": hedge_2_profile["price"],
        "ytm": hedge_2_profile["ytm"],
        "dv01": hedge_2_profile["dv01"],
        "duration": hedge_2_profile["duration"],
        "convexity": hedge_2_profile["convexity"]
    }
])

print("Instrument risk profiles:")
print(q11_instruments_table.to_string(index=False))

# 11a) DV01 hedge with bond 1
hedge_value_dv01 = -(open_position_value * open_profile["dv01"] / hedge_1_profile["dv01"])

# 11b) Duration hedge with bond 2
hedge_value_duration = - (open_profile["duration"] * open_position_value) / hedge_2_profile["duration"]

# 11c) Duration-convexity hedge with both bonds
a11 = hedge_1_profile["duration"]
a12 = hedge_2_profile["duration"]
b1 = -(open_position_value * open_profile["duration"])

a21 = hedge_1_profile["convexity"]
a22 = hedge_2_profile["convexity"]
b2 = -(open_position_value * open_profile["convexity"])

det = a11 * a22 - a12 * a21
if abs(det) < 1e-14:
    raise ValueError("The 2x2 hedge system is singular.")

Vh1 = (b1 * a22 - a12 * b2) / det
Vh2 = (a11 * b2 - b1 * a21) / det

q11a_table = pd.DataFrame([
    {"metric": "Open position market value (EUR)", "value": open_position_value},
    {"metric": "Required hedge value using DV01 (EUR)", "value": hedge_value_dv01},
    {"metric": "Recommended position", "value": "SHORT hedge bond 1" if hedge_value_dv01 < 0 else "LONG hedge bond 1"},
])

q11b_table = pd.DataFrame([
    {"metric": "Required hedge value using duration (EUR)", "value": hedge_value_duration},
    {"metric": "Recommended position", "value": "SHORT hedge bond 2" if hedge_value_duration < 0 else "LONG hedge bond 2"},
])

q11c_table = pd.DataFrame([
    {"metric": "Required value in hedge bond 1 (EUR)", "value": Vh1},
    {"metric": "Position in hedge bond 1", "value": "SHORT" if Vh1 < 0 else "LONG"},
    {"metric": "Required value in hedge bond 2 (EUR)", "value": Vh2},
    {"metric": "Position in hedge bond 2", "value": "SHORT" if Vh2 < 0 else "LONG"},
])

print("\n11a) Hedging strategy with the first bond using DV01")
print(q11a_table.to_string(index=False))

print("\n11b) Hedging strategy with the second bond using duration")
print(q11b_table.to_string(index=False))

print("\n11c) Hedging strategy with two bonds using duration and convexity")
print(q11c_table.to_string(index=False))

q11_instruments_table.to_csv("question_11_instrument_profiles.csv", index=False)
q11a_table.to_csv("question_11a_dv01_hedge.csv", index=False)
q11b_table.to_csv("question_11b_duration_hedge.csv", index=False)
q11c_table.to_csv("question_11c_duration_convexity_hedge.csv", index=False)

print("\nInterpretation:")
print(
    "A DV01 hedge neutralises the euro sensitivity to a 1 basis point yield move. "
    "A duration hedge neutralises first-order proportional sensitivity, but not convexity. "
    "A two-bond duration-convexity hedge is more robust because it neutralises both first-order "
    "and second-order exposure."
)


# ============================================================
# FINAL DELIVERABLE SUMMARY
# ============================================================
final_summary = pd.DataFrame([
    {"question": "Q2", "metric": "ECB date used", "value": actual_date_used},
    {"question": "Q4", "metric": "Interpolation error (bps)", "value": interpolation_error_bp},
    {"question": "Q7a", "metric": "Bond price using spot rates", "value": price_q7_spot},
    {"question": "Q7b", "metric": "Bond price using forward rates", "value": price_q7_forward},
    {"question": "Q8", "metric": "Dirty price", "value": dirty_q8},
    {"question": "Q8", "metric": "Clean price", "value": clean_q8},
    {"question": "Q9", "metric": "YTM", "value": ytm_q9},
    {"question": "Q10a", "metric": "DV01", "value": dv01_q10a},
    {"question": "Q10b", "metric": "Duration", "value": duration_q10b},
    {"question": "Q10b", "metric": "Convexity", "value": convexity_q10b},
    {"question": "Q11a", "metric": "DV01 hedge value", "value": hedge_value_dv01},
    {"question": "Q11b", "metric": "Duration hedge value", "value": hedge_value_duration},
    {"question": "Q11c", "metric": "Duration-convexity hedge value bond 1", "value": Vh1},
    {"question": "Q11c", "metric": "Duration-convexity hedge value bond 2", "value": Vh2},
])

print_section("FINAL SUMMARY")
print(final_summary.to_string(index=False))
final_summary.to_csv("final_project_summary.csv", index=False)

print("\nFiles created:")
print("- ecb_aaa_spot_curve_selected_date.csv")
print("- question_2_spot_yields.csv")
print("- question_3_spot_curve.png")
print("- question_5_strips_prices.csv")
print("- question_6_forward_rates.csv")
print("- question_6_forward_curve.png")
print("- question_7_bond_prices.csv")
print("- question_8_dirty_clean_price.csv")
print("- question_9_ytm.csv")
print("- question_10a_dv01.csv")
print("- question_10b_duration_convexity.csv")
print("- question_10c_large_shock.csv")
print("- question_11_instrument_profiles.csv")
print("- question_11a_dv01_hedge.csv")
print("- question_11b_duration_hedge.csv")
print("- question_11c_duration_convexity_hedge.csv")
print("- final_project_summary.csv")
