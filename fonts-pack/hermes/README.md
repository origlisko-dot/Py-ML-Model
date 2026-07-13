# Hermès Fonts Pack

An open-source typography pack inspired by the **Hermès** luxury brand logo — the iconic slab-serif wordmark with wide letter-spacing and signature orange.

> **Note:** The official Hermès logo uses a custom typeface based on **Memphis Bold** (Linotype). This pack provides legally free, open-source alternatives that closely match the geometric slab-serif aesthetic.

## What's Included

| Font | Role | Weights | License |
|------|------|---------|---------|
| **Sanchez** | Primary (closest to Memphis) | 400, 400 Italic | [OFL 1.1](https://scripts.sil.org/OFL) |
| **Roboto Slab** | Secondary / bold headlines | 300–900 | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| **Arvo** | Display alternative | 400, 700 + italics | [OFL 1.1](https://scripts.sil.org/OFL) |
| **Zilla Slab** | Body / accent text | 300–700 + italics | [OFL 1.1](https://scripts.sil.org/OFL) |

### Files

```
fonts-pack/hermes/
├── fonts/           # WOFF2 font files
│   ├── sanchez/
│   ├── roboto-slab/
│   ├── arvo/
│   └── zilla-slab/
├── css/
│   ├── fonts.css        # @font-face declarations
│   ├── tokens.css       # Brand color & typography variables
│   └── hermes-pack.css  # Main entry (import this)
├── preview/
│   └── index.html       # Live specimen page
├── assets/
│   └── carriage.svg     # Decorative carriage icon
├── LICENSES.md
└── README.md
```

## Quick Start

### Web (HTML)

```html
<link rel="stylesheet" href="fonts-pack/hermes/css/hermes-pack.css">

<div class="hermes-lockup">
  <h1 class="hermes-logo">HERMÈS</h1>
  <p class="hermes-subtitle">PARIS</p>
</div>
```

### CSS Variables

```css
@import url("fonts-pack/hermes/css/hermes-pack.css");

.my-heading {
  font-family: var(--hermes-font-primary);
  color: var(--hermes-orange);
  letter-spacing: var(--hermes-tracking-wide);
  text-transform: uppercase;
}
```

### Preview

Open `preview/index.html` in a browser to see all specimens, weights, and the brand palette.

```bash
# Serve locally (optional)
python3 -m http.server 8080 --directory fonts-pack/hermes/preview
```

## Typography Characteristics

The Hermès wordmark is defined by:

- **Slab serifs** — thick, rectangular serifs (Egyptian / geometric slab style)
- **Monoline strokes** — uniform thickness, no contrast between thick and thin
- **Wide tracking** — generous letter-spacing (~0.25–0.30em)
- **All caps** — uppercase only in the logo lockup
- **Signature orange** — `#F37021`

### Official vs. This Pack

| Aspect | Official Hermès | This Pack |
|--------|----------------|-----------|
| Typeface | Memphis Bold (custom) | Sanchez (primary) |
| License | Proprietary | Open source (OFL / Apache) |
| Use | Brand only | Personal & commercial projects |

## Utility Classes

| Class | Description |
|-------|-------------|
| `.hermes-lockup` | Centered logo container |
| `.hermes-logo` | Main wordmark style |
| `.hermes-subtitle` | "PARIS" subline style |
| `.hermes-display` | Bold display headings |
| `.hermes-body` | Body copy |
| `.hermes-caption` | Small caps captions |
| `.hermes-orange` | Brand orange text color |
| `.hermes-bg-cream` | Cream background |

## Brand Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `--hermes-orange` | `#F37021` | Primary brand color |
| `--hermes-orange-dark` | `#D45F18` | Hover / emphasis |
| `--hermes-orange-light` | `#FF8A3D` | Highlights |
| `--hermes-cream` | `#FAF7F2` | Backgrounds |
| `--hermes-ink` | `#1A1A1A` | Body text |
| `--hermes-muted` | `#6B6B6B` | Secondary text |

## Disclaimer

This pack is **not affiliated with or endorsed by Hermès International S.A.** It is an independent typography resource for designers seeking a similar aesthetic using freely licensed fonts. Do not use this pack to impersonate the Hermès brand.
