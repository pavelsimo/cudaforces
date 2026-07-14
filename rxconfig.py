import os

import reflex as rx

config = rx.Config(
    app_name="cudaforces",
    api_url=os.getenv("REFLEX_API_URL", "http://localhost:8000"),
    telemetry_enabled=False,
    plugins=[
        rx.plugins.RadixThemesPlugin(theme=rx.theme(appearance="dark", accent_color="green")),
        rx.plugins.SitemapPlugin(),
    ],
)
