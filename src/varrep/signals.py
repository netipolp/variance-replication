import pandas as pd


def compute_trading_signals(results_df, ema_span=10, std_span=10, volatility_span=5,
                            bb_switch=1.5, volatility_benchmark=0.01):
    """
        Requires column: results_df['price'].
    """
    # results_df = results_df.copy().reset_index(drop=True)

    results_df['EMA'] = results_df['price'].ewm(span=ema_span, adjust=False).mean()
    results_df['std'] = results_df['price'].rolling(window=std_span).std()
    results_df['upper_band'] = results_df['EMA'] + bb_switch * results_df['std']
    results_df['lower_band'] = results_df['EMA'] - bb_switch * results_df['std']
    results_df['vol_diff'] = results_df['price'].diff().abs()
    results_df['vol_EMA'] = results_df['vol_diff'].ewm(span=volatility_span, adjust=False).mean()

    results_df['signal2'] = 0
    results_df['switch'] = 0

    switch = 0  # -1 short, 0 flat, +1 long

    for i in range(1, len(results_df)):
        vol_today = results_df.loc[i, 'price']
        EMA = results_df.loc[i, 'EMA']
        upper_band = results_df.loc[i, 'upper_band']
        lower_band = results_df.loc[i, 'lower_band']
        vol_vol = results_df.loc[i, 'vol_EMA']

        if (switch == 1 and vol_today >= EMA) or (switch == -1 and vol_today <= EMA):
            switch = 0

        # matches notebook: <= benchmark
        if switch == 0 and vol_vol <= volatility_benchmark:
            if vol_today <= lower_band:
                switch = 1
            elif vol_today >= upper_band:
                switch = -1

        results_df.loc[i, 'switch'] = switch
        results_df.loc[i, 'signal2'] = switch

    return results_df
