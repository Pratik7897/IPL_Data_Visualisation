# nginx/ssl/

This directory is populated at runtime by Certbot (Let's Encrypt).

**Do NOT commit SSL certificates to git.**

After running `scripts/setup_ssl.sh`, certs will be placed here at:

```
nginx/ssl/live/YOUR_DOMAIN/
  ├── fullchain.pem   ← certificate + intermediate chain
  ├── privkey.pem     ← private key
  ├── cert.pem        ← certificate only
  └── chain.pem       ← intermediate chain only
```

The Nginx vhost config references `fullchain.pem` and `privkey.pem`.
