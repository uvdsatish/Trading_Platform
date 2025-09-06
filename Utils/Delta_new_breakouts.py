import sys
from breakout_scanner import check_52week_breakouts
from breakout_scanner_25day import check_25day_breakouts
from breakout_scanner_100day import check_100day_breakouts

# Your ticker list

my_tickers = [
    # Longs delta
     "AG", "AGX", "AMSC", "APP", "ARKF", "ARKK", "ARKQ", "ARKW", "ARKX",
    "CALM", "CCEC", "COHR", "CRPT", "CVNA", "DASH", "DAVE", "EPOL", "FIX",
    "FPX", "FTI", "GEV", "GREK", "HOOD", "IBIT", "IZRL", "JOYY", "KB", "KURE",
    "NPBI", "PBI", "PRA", "PSIX", "PWRD", "QTUM", "SBSW", "SEZL", "SGHC",
    "SHOC", "SITM", "THG", "TOST", "TSSI", "TTMI", "UBER", "URA", "URBN",
    "UTES", "VIRT", "XAR", "YSG",

    # Shorts delta
    "CABO", "CGON", "CRNX", "GPCR", "HELE", "JANX", "KIDS", "MAGN", "OLN", "RARE", "RVMD", "SNDX", "USPH", "VERA",
    "WEST", "WGO", "ZIP"
]
results_yearly = check_52week_breakouts(my_tickers)

results_25 = check_25day_breakouts(my_tickers)

results_100 = check_100day_breakouts(my_tickers)

sys.exit(0)

