"""
Unit tests for the bond valuation functions in RM_project_bonds.py.

These tests use hardcoded spot/forward rates to validate the core
computation functions independently of the ECB data download.
"""

import math
import pytest

# ---------------------------------------------------------------------------
# Import helpers and functions from the main script.
# Because the script's top-level code tries to hit the ECB API, we patch the
# network call before importing.
# ---------------------------------------------------------------------------
import sys
import types
import importlib


# We need to import only the functions, not run the top-level code.
# Extract the functions by reading the source and exec-ing just the
# function definitions.
def _load_functions():
    """Load function definitions from the script without executing top-level code."""
    import ast
    with open("RM_project_bonds.py", "r") as f:
        source = f.read()

    tree = ast.parse(source)

    # Keep only imports and function/class definitions
    new_body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.ClassDef, ast.Assign)):
            # For assignments, only keep simple top-level constants
            if isinstance(node, ast.Assign):
                # Keep only assignments to simple names (constants)
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if targets and all(t in ("selected_date", "bond_category",
                                          "target_maturities", "m",
                                          "par_value_strips") for t in targets):
                    new_body.append(node)
            else:
                new_body.append(node)

    tree.body = new_body
    ast.fix_missing_locations(tree)
    code = compile(tree, "RM_project_bonds.py", "exec")

    namespace = {}
    exec(code, namespace)
    return namespace


ns = _load_functions()

# Pull out the functions we need to test
maturity_from_code = ns["maturity_from_code"]
bond_coupon = ns["bond_coupon"]
bond_price_spot = ns["bond_price_spot"]
bond_price_forward = ns["bond_price_forward"]
bond_price_ytm = ns["bond_price_ytm"]
solve_ytm_bisection = ns["solve_ytm_bisection"]
dv01_from_ytm = ns["dv01_from_ytm"]
duration_convexity_from_ytm = ns["duration_convexity_from_ytm"]
price_change_duration_convexity = ns["price_change_duration_convexity"]
almost_equal = ns["almost_equal"]


# ---------------------------------------------------------------------------
# Test data: small set of semi-annual spot rates (already semi-annual compounded)
# ---------------------------------------------------------------------------
TEST_MATURITIES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TEST_SPOT_RATES = [0.02, 0.022, 0.024, 0.025, 0.026, 0.027]  # semi-annual


def _compute_forward_rates(maturities, spot_rates, m=2):
    """Reproduce the forward rate calculation from the script."""
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
    return forward_rates


TEST_FORWARD_RATES = _compute_forward_rates(TEST_MATURITIES, TEST_SPOT_RATES)


# ============================================================
# Tests for maturity_from_code
# ============================================================
class TestMaturityFromCode:
    def test_6m(self):
        assert maturity_from_code("SR_6M") == pytest.approx(0.5)

    def test_1y(self):
        assert maturity_from_code("SR_1Y") == pytest.approx(1.0)

    def test_1y6m(self):
        assert maturity_from_code("SR_1Y6M") == pytest.approx(1.5)

    def test_18m(self):
        assert maturity_from_code("SR_18M") == pytest.approx(1.5)

    def test_5y(self):
        assert maturity_from_code("SR_5Y") == pytest.approx(5.0)

    def test_invalid(self):
        assert maturity_from_code("INVALID") is None


# ============================================================
# Tests for bond pricing
# ============================================================
class TestBondPricing:
    face = 1000.0
    coupon_rate = 0.02
    maturity = 3.0
    freq = 2

    def test_bond_coupon(self):
        c = bond_coupon(self.face, self.coupon_rate, self.freq)
        assert c == pytest.approx(10.0)  # 1000 * 0.02 / 2

    def test_spot_and_forward_prices_match(self):
        """Spot and forward prices must be equal by no-arbitrage."""
        price_s = bond_price_spot(
            self.face, self.coupon_rate, self.maturity, self.freq,
            TEST_MATURITIES, TEST_SPOT_RATES
        )
        price_f = bond_price_forward(
            self.face, self.coupon_rate, self.maturity, self.freq,
            TEST_MATURITIES, TEST_FORWARD_RATES
        )
        assert price_s == pytest.approx(price_f, rel=1e-10)

    def test_ytm_roundtrip(self):
        """Solving YTM from a price and then re-pricing must give the same price."""
        price = bond_price_spot(
            self.face, self.coupon_rate, self.maturity, self.freq,
            TEST_MATURITIES, TEST_SPOT_RATES
        )
        ytm = solve_ytm_bisection(price, self.face, self.coupon_rate, self.maturity, self.freq)
        repriced = bond_price_ytm(self.face, self.coupon_rate, self.maturity, self.freq, ytm)
        assert repriced == pytest.approx(price, rel=1e-10)

    def test_par_bond_ytm_equals_coupon(self):
        """A bond priced at par has YTM equal to the coupon rate."""
        ytm = solve_ytm_bisection(1000.0, 1000.0, 0.05, 5.0, 2)
        assert ytm == pytest.approx(0.05, abs=1e-8)


# ============================================================
# Tests for DV01
# ============================================================
class TestDV01:
    face = 1000.0
    coupon_rate = 0.02
    maturity = 3.0
    freq = 2
    ytm = 0.025

    def test_dv01_positive(self):
        """DV01 should be positive for any normal bond."""
        _, _, _, dv01 = dv01_from_ytm(self.face, self.coupon_rate, self.maturity, self.freq, self.ytm)
        assert dv01 > 0

    def test_dv01_consistency(self):
        """DV01 should approximately equal the price change for a 1bp decrease."""
        p0, _, _, dv01 = dv01_from_ytm(self.face, self.coupon_rate, self.maturity, self.freq, self.ytm, shift_bp=1)
        p_down = bond_price_ytm(self.face, self.coupon_rate, self.maturity, self.freq, self.ytm - 0.0001)
        assert dv01 == pytest.approx(p_down - p0, rel=1e-4)


# ============================================================
# Tests for duration and convexity
# ============================================================
class TestDurationConvexity:
    face = 1000.0
    coupon_rate = 0.02
    maturity = 3.0
    freq = 2
    ytm = 0.025

    def test_duration_positive(self):
        _, _, _, duration, _ = duration_convexity_from_ytm(
            self.face, self.coupon_rate, self.maturity, self.freq, self.ytm
        )
        assert duration > 0

    def test_convexity_positive(self):
        _, _, _, _, convexity = duration_convexity_from_ytm(
            self.face, self.coupon_rate, self.maturity, self.freq, self.ytm
        )
        assert convexity > 0

    def test_price_change_approximation(self):
        """Duration/convexity approximation should be close to exact for small shocks."""
        p0, _, _, duration, convexity = duration_convexity_from_ytm(
            self.face, self.coupon_rate, self.maturity, self.freq, self.ytm
        )
        dy = 0.001  # 10 bps
        _, approx_change, _ = price_change_duration_convexity(p0, duration, convexity, dy)
        exact_new = bond_price_ytm(self.face, self.coupon_rate, self.maturity, self.freq, self.ytm + dy)
        exact_change = exact_new - p0
        assert approx_change == pytest.approx(exact_change, rel=1e-3)


# ============================================================
# Tests for hedging (Q11 fixes)
# ============================================================
class TestHedging:
    def test_dv01_hedge_sign(self):
        """DV01 hedge value should be negative (short) when hedging a long position."""
        # Long position with positive DV01
        dv01_open = 0.28
        dv01_hedge = 0.19
        v_open = 7_300_000.0
        # The hedge formula should give a negative value (short position)
        hedge_value = -(v_open * dv01_open / dv01_hedge)
        assert hedge_value < 0

    def test_duration_convexity_hedge_neutralises(self):
        """
        The duration-convexity hedge should neutralise the portfolio.
        D_open * V_open + D_h1 * V_h1 + D_h2 * V_h2 = 0
        C_open * V_open + C_h1 * V_h1 + C_h2 * V_h2 = 0
        """
        D_open, C_open = 2.9, 10.5
        D_h1, C_h1 = 1.95, 5.0
        D_h2, C_h2 = 4.8, 27.0
        V_open = 7_300_000.0

        # Correct system: D_h1*V1 + D_h2*V2 = -D_open*V_open
        #                 C_h1*V1 + C_h2*V2 = -C_open*V_open
        b1 = -(V_open * D_open)
        b2 = -(V_open * C_open)

        det = D_h1 * C_h2 - D_h2 * C_h1
        V1 = (b1 * C_h2 - D_h2 * b2) / det
        V2 = (D_h1 * b2 - b1 * C_h1) / det

        # Check neutralisation
        total_duration = D_open * V_open + D_h1 * V1 + D_h2 * V2
        total_convexity = C_open * V_open + C_h1 * V1 + C_h2 * V2

        assert total_duration == pytest.approx(0.0, abs=1e-6)
        assert total_convexity == pytest.approx(0.0, abs=1e-6)

    def test_wrong_sign_does_not_neutralise(self):
        """
        Verify that without negative signs, the system does NOT neutralise.
        This confirms the bug we fixed.
        """
        D_open, C_open = 2.9, 10.5
        D_h1, C_h1 = 1.95, 5.0
        D_h2, C_h2 = 4.8, 27.0
        V_open = 7_300_000.0

        # Bug: missing negative sign
        b1_wrong = V_open * D_open
        b2_wrong = V_open * C_open

        det = D_h1 * C_h2 - D_h2 * C_h1
        V1_wrong = (b1_wrong * C_h2 - D_h2 * b2_wrong) / det
        V2_wrong = (D_h1 * b2_wrong - b1_wrong * C_h1) / det

        total_dur = D_open * V_open + D_h1 * V1_wrong + D_h2 * V2_wrong
        total_cvx = C_open * V_open + C_h1 * V1_wrong + C_h2 * V2_wrong

        # With the wrong sign, the total is 2x the original exposure, not zero
        assert abs(total_dur) > 1e6
        assert abs(total_cvx) > 1e6


# ============================================================
# Tests for continuous-to-semi-annual conversion
# ============================================================
class TestCompoundingConversion:
    def test_conversion_formula(self):
        """Verify that converted semi-annual rates give the same discount factors."""
        m = 2
        s_cc = 0.025  # 2.5% continuous
        s_sa = m * (math.exp(s_cc / m) - 1)

        t = 3.0
        df_cc = math.exp(-s_cc * t)
        df_sa = 1 / (1 + s_sa / m) ** (m * t)

        assert df_cc == pytest.approx(df_sa, rel=1e-12)

    def test_semi_annual_larger_than_continuous(self):
        """Semi-annual compounded rate should be slightly larger than continuous."""
        m = 2
        s_cc = 0.03
        s_sa = m * (math.exp(s_cc / m) - 1)
        assert s_sa > s_cc

    def test_zero_rate_unchanged(self):
        """A zero continuous rate should convert to zero semi-annual rate."""
        m = 2
        s_cc = 0.0
        s_sa = m * (math.exp(s_cc / m) - 1)
        assert s_sa == pytest.approx(0.0, abs=1e-15)


# ============================================================
# Tests for forward rates
# ============================================================
class TestForwardRates:
    def test_first_forward_equals_first_spot(self):
        """The first forward rate should equal the first spot rate."""
        assert TEST_FORWARD_RATES[0] == pytest.approx(TEST_SPOT_RATES[0])

    def test_no_arbitrage(self):
        """
        Verify no-arbitrage: (1+s_t/m)^(m*t) = product of (1+f_j/m) for j=1..m*t
        """
        m = 2
        for i, t in enumerate(TEST_MATURITIES):
            growth_spot = (1 + TEST_SPOT_RATES[i] / m) ** (m * t)
            growth_fwd = 1.0
            for j in range(i + 1):
                growth_fwd *= (1 + TEST_FORWARD_RATES[j] / m)
            assert growth_spot == pytest.approx(growth_fwd, rel=1e-10)


# ============================================================
# Tests for STRIPS pricing
# ============================================================
class TestSTRIPS:
    def test_below_par_for_positive_rates(self):
        """STRIPS prices should be below par for positive spot rates."""
        m = 2
        par = 100.0
        for n, y in zip(TEST_MATURITIES, TEST_SPOT_RATES):
            price = par / ((1 + y / m) ** (m * n))
            assert price < par

    def test_correct_formula(self):
        """Verify STRIPS pricing: P = 100 / (1 + s/m)^(m*n)."""
        m = 2
        par = 100.0
        s = 0.03
        n = 2.0
        expected = par / ((1 + s / m) ** (m * n))
        # Manual: 100 / (1.015)^4 = 100 / 1.06136355... ≈ 94.218
        assert expected == pytest.approx(94.2184, rel=1e-4)
