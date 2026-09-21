# VPN Setup Guide

All employees must connect through the Nimbus VPN when working remotely.

## Installing the Client

Download **NimbusConnect** from the IT portal. Supported on Windows 11 and macOS 14+.

Run the installer and sign in with your SSO account.

## Troubleshooting

### Error ERR_TUNNEL_4012

This error means your device certificate has expired. Open NimbusConnect,
go to `Settings > Certificates`, and click **Renew**.

```
# Do not confuse this with a heading
nimbus-vpn --renew-cert
```

### Error ERR_TUNNEL_4013

The gateway rejected your region. Open `Settings > Gateway` and pick the
gateway for the country you are working from.

### Slow connection

Switch the gateway to the region closest to you, for example `eu-west-gw`.
