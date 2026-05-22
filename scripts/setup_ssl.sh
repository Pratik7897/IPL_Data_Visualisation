#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup_ssl.sh  —  Phase 3B Let's Encrypt SSL setup
#
# Run this ONCE on your server after DNS is pointed at the server IP.
# Pre-requisites:
#   1. Docker + Docker Compose installed
#   2. Ports 80 & 443 open in firewall
#   3. DNS A record pointing YOUR_DOMAIN → server IP
#
# Usage:
#   chmod +x scripts/setup_ssl.sh
#   ./scripts/setup_ssl.sh YOUR_DOMAIN YOUR_EMAIL
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-}"

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
  echo "❌  Usage: $0 <domain> <email>"
  echo "    Example: $0 ipl.example.com admin@example.com"
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  IPL Dashboard — Let's Encrypt SSL Setup                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Domain : $DOMAIN"
echo "  Email  : $EMAIL"
echo ""

# ── Step 1: Make sure Nginx is running in HTTP-only mode ──────────────────────
echo "▶  Step 1/4  Starting Nginx (HTTP-only mode) …"
docker compose up -d nginx
sleep 3
echo "   Nginx started. Testing config …"
docker compose exec nginx nginx -t && echo "   ✅  Nginx config valid"

# ── Step 2: Issue certificate via ACME webroot challenge ──────────────────────
echo ""
echo "▶  Step 2/4  Issuing Let's Encrypt certificate for $DOMAIN …"
docker compose --profile ssl run --rm certbot certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  --domain "$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos \
  --non-interactive \
  --rsa-key-size 4096 \
  && echo "   ✅  Certificate issued!"

# ── Step 3: Activate SSL in the Nginx vhost config ────────────────────────────
echo ""
echo "▶  Step 3/4  Activating HTTPS block in Nginx config …"
CONF="nginx/conf.d/ipl_dashboard.conf"

# Replace placeholder domain
sed -i.bak "s/YOUR_DOMAIN/$DOMAIN/g" "$CONF"

# Uncomment the HTTPS server block (remove leading '# ' from the block)
python3 - <<'PYEOF'
import re, sys
path = "nginx/conf.d/ipl_dashboard.conf"
with open(path) as f:
    content = f.read()

# Remove '# ' prefix from the commented-out HTTPS server block
# Find the block between the two marker comments
content = re.sub(
    r'^# (server \{.*?^# \})',
    lambda m: m.group(1).replace('\n# ', '\n'),
    content,
    flags=re.MULTILINE | re.DOTALL,
)
with open(path, "w") as f:
    f.write(content)
print("   HTTPS block uncommented.")
PYEOF

# Also enable the HTTP → HTTPS redirect in the HTTP block
sed -i.bak 's|# return 301|return 301|' "$CONF"
# Comment out the HTTP proxy location (now redirects instead)
sed -i.bak '/# Proxy to Dash (HTTP-only mode)/,/^    \}/s/^/    # /' "$CONF"
echo "   ✅  HTTPS block activated, HTTP redirects to HTTPS."

# ── Step 4: Reload Nginx with new SSL config ──────────────────────────────────
echo ""
echo "▶  Step 4/4  Reloading Nginx with SSL config …"
docker compose exec nginx nginx -t && \
docker compose exec nginx nginx -s reload
echo "   ✅  Nginx reloaded!"

# ── Start Certbot auto-renewal ─────────────────────────────────────────────────
echo ""
echo "▶  Starting Certbot auto-renewal daemon …"
docker compose --profile ssl up -d certbot
echo "   ✅  Certbot renewal scheduled (every 12h)."

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅  SSL Setup Complete!                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Dashboard  : https://$DOMAIN"
echo "  HTTP       : http://$DOMAIN  → redirects to HTTPS"
echo ""
echo "  Test SSL grade:  https://www.ssllabs.com/ssltest/analyze.html?d=$DOMAIN"
echo ""
