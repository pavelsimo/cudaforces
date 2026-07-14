"""Design tokens from the CudaForces mockup (example/CudaForces.dc.html)."""

BG = "#0c0f12"
PANEL = "#0d1117"
CARD = "#11161c"
HEADING = "#f0f4f8"
TEXT = "#dbe2ea"
MUTED = "#8b98a5"
FAINT = "#78848f"
DIM = "#57626e"
ACCENT = "#7ee787"
ACCENT_HOVER = "#a5f3ab"
ACCENT_BG = "rgba(126,231,135,.05)"
ACCENT_BORDER = "rgba(126,231,135,.35)"
BORDER = "rgba(255,255,255,.08)"
BORDER_STRONG = "rgba(255,255,255,.12)"
HOVER_BG = "rgba(255,255,255,.03)"

FONT = "'Space Grotesk', 'Helvetica Neue', system-ui, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace"
FONTS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Space+Grotesk:wght@400;500;600;700"
    "&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400"
    "&display=swap"
)

DIFFICULTY_COLORS = {"Easy": "#56d364", "Medium": "#e3b341", "Hard": "#f85149"}

VERDICT_COLORS = {
    "AC": "#56d364",
    "WA": "#f85149",
    "CE": "#e3b341",
    "RE": "#f85149",
    "TLE": "#e3b341",
    "JE": "#8b98a5",
}

VERDICT_LABELS = {
    "AC": "Accepted",
    "WA": "Wrong Answer",
    "CE": "Compile Error",
    "RE": "Runtime Error",
    "TLE": "Time Limit Exceeded",
    "JE": "Judge Error",
}


def difficulty_color(difficulty: str) -> str:
    return DIFFICULTY_COLORS.get(difficulty, MUTED)


def rating_color(rating: int) -> str:
    if rating < 1200:
        return "#9aa4ae"
    if rating < 1400:
        return "#56d364"
    if rating < 1600:
        return "#4dd0c4"
    if rating < 1900:
        return "#58a6ff"
    if rating < 2100:
        return "#bc8cff"
    if rating < 2400:
        return "#ffa657"
    return "#f85149"
