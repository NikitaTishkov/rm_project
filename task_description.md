# Project - Valuation and Hedging – Bonds & Yields

## Part 1

1. Define a bond and explain how it is issued. Explain the relationship between bonds and interest rates.

2. From the website of the European Central Bank – "Euro area yield curves" section:
   - Select a specific date for the AAA rated bonds (or for all bonds).
   - For the selected date, choose from the spot yield curve 10 consecutive semi-annual spot yields (The yields are those of zero-coupon bonds, with a par value of €100. 10 consecutive semi-annual spot yields represent a period of 5 years.):
     https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html

3. Define a spot rate and its characteristics. Explain the method through which spot rates are derived from the market. Plot the spot yield curve. Interpret from a macroeconomic perspective.

4. Using two non-consecutive semi-annual spot yields (for example, n1 and n3, where "n" defines the 10 semi-annual spot yields you have selected), perform a linear interpolation to estimate the value of another semi-annual spot yield (for example, n2). Compare the estimated value of the semi-annual spot yield to the observed one. Interpret.

5. Calculate the price of the 10 STRIPS from which the chosen 10 spot rates are deducted by reversing the equation on slide 21. Are the STRIPS below or above par value? Why?

6. Define a forward rate and its characteristics. Explain how forward rates are derived from the market. From the chosen spot rates, calculate the respective forward rates. Plot the forward yield curve.

## Part 2

7. Assume that today is the 1st of January 2025. Assume that the French government issued today (1st of January 2025) a 3-year government bond that pays semi-annually a 2% coupon (par value = 1,000€).
   - a) Calculate the price of the bond using spot rates.
   - b) Calculate the price of the bond using forward rates.
   - c) What do you observe?

8. Assume that you have purchased the 3-year government bond 50 days after it was issued. Using the spot rates, calculate the dirty and the clean price of the bond.

9. Estimate the yield to maturity (YTM) of the bond calculated in question 5 a). Interpret.

10. Using the bond parameters calculated in question 7 a) and the YTM estimated in question 9:
    - a) Calculate DV01 for a five basis points decrease in YTM. Interpret.
    - b) Calculate duration and convexity for a 0.55% increase and decrease in YTM. Interpret.
    - c) Calculate the change in bond price for a 1.8% increase and decrease in YTM using the duration/convexity approach. Interpret.

## Part 3

11. Choose the DV01, duration and convexity measures of two bonds (either randomly / either from a real economy) to create hedging strategies. Assume you decided to purchase $7.3M of the bond calculated in question 7 a):
    - a) Create a hedging strategy with the first bond using the DV01 measure. Interpret.
    - b) Create a hedging strategy with the second bond using the duration measure. Interpret.
    - c) Create a hedging strategy with the two bonds using the duration/convexity approach. Interpret.

## Assignment Details

- **Deliverables:** Dataset + Python Jupyter Notebook
- **Deadline:** Sunday - 5th of April, 21h
- **Note:** Write the equations as shown in the slides, do not use Python libraries
