# cTrader Data Source Setup

This guide covers obtaining the four credentials required by `CTraderDataSource`:
`client_id`, `client_secret`, `access_token`, `account_id`.

---

## 1. Get Client ID and Secret

1. Go to [connect.spotware.com/apps](https://connect.spotware.com/apps) and log in with your cTrader account.
2. Create a new application (or open an existing one).
3. Copy **Client ID** and **Client Secret** from the app settings.
4. Set a redirect URI — for local use `https://localhost` is fine.

---

## 2. Get an Access Token (OAuth 2.0)

### Step 1 — Authorize in the browser

Open the following URL, substituting your values:

```
https://connect.spotware.com/apps/auth?client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT_URI&scope=trading&response_type=code
```

Log in and authorize. The browser will redirect to your redirect URI with a `code` query parameter:

```
https://localhost?code=ABC123...
```

Copy that code — it expires in a few minutes.

### Step 2 — Exchange the code for a token

```bash
curl -X POST https://connect.spotware.com/apps/token \
  -d "grant_type=authorization_code" \
  -d "code=ABC123..." \
  -d "redirect_uri=YOUR_REDIRECT_URI" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET"
```

The response contains `access_token` and `refresh_token`:

```json
{
  "access_token": "ey...",
  "refresh_token": "ey...",
  "token_type": "bearer",
  "expires_in": 2592000
}
```

Save the `access_token`. When it expires, repeat this step using `grant_type=refresh_token&refresh_token=YOUR_REFRESH_TOKEN` instead.

---

## 3. Get Account ID

Run this script once to print all cTrader accounts linked to your access token:

```python
import asyncio, ssl
from trading.data.ctrader_datasource import _send, _recv_type
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq, ProtoOAApplicationAuthRes,
    ProtoOAGetAccountListByAccessTokenReq, ProtoOAGetAccountListByAccessTokenRes,
)

async def get_accounts(client_id: str, client_secret: str, access_token: str) -> None:
    ssl_ctx = ssl.create_default_context()
    reader, writer = await asyncio.open_connection("live.ctraderapi.com", 5035, ssl=ssl_ctx)

    req = ProtoOAApplicationAuthReq()
    req.clientId = client_id
    req.clientSecret = client_secret
    await _send(writer, req)
    await _recv_type(reader, ProtoOAApplicationAuthRes)

    req2 = ProtoOAGetAccountListByAccessTokenReq()
    req2.accessToken = access_token
    await _send(writer, req2)
    res = await _recv_type(reader, ProtoOAGetAccountListByAccessTokenRes)

    for acc in res.ctidTraderAccount:
        kind = "live" if acc.isLive else "demo"
        print(f"Account ID: {acc.ctidTraderAccountId}  ({kind})")

    writer.close()

asyncio.run(get_accounts("YOUR_CLIENT_ID", "YOUR_CLIENT_SECRET", "YOUR_ACCESS_TOKEN"))
```

```
uv run python scripts/get_ctrader_accounts.py
```

---

## 4. Add to .env

```env
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=your_account_id
```

---

## 5. Usage

```python
import os
from trading.data import CTraderDataSource

source = CTraderDataSource(
    client_id=os.environ["CTRADER_CLIENT_ID"],
    client_secret=os.environ["CTRADER_CLIENT_SECRET"],
    access_token=os.environ["CTRADER_ACCESS_TOKEN"],
    account_id=int(os.environ["CTRADER_ACCOUNT_ID"]),
)

df = source.get_ohlcv("EURUSD", "1h", 100)
df = source.get_ohlcv("XAUUSD", "4h", 72)   # Gold
df = source.get_ohlcv("US500", "1d", 30)     # S&P 500 CFD
```

Supported timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `12h`, `1d`, `1w`.

Symbol names follow cTrader conventions (e.g. `EURUSD`, `XAUUSD`, `BTCUSD`, `US500`) — not the `BTC/USDT` format used by Binance.

---

## Demo vs Live

To use a demo account, point to the demo host:

```python
source = CTraderDataSource(
    ...,
    host="demo.ctraderapi.com",
)
```

Demo account IDs are obtained the same way — they appear in the account list with `isLive=False`.
