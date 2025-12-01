import reflex as rx
import json
from typing import List

class State(rx.State):
    status: str = "IDLE"
    confidence: float = 0.0
    risk_factors: List[str] = []
    is_streaming: bool = False
    status_color: str = "gray"

    def toggle_stream(self):
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            # Connects to localhost:8000
            return rx.call_script("startStream('ws://localhost:8001/ws/stream')")
        else:
            self.status = "IDLE"
            self.confidence = 0.0
            self.risk_factors = []
            self.status_color = "gray"
            return rx.call_script("stopStream()")

    def handle_bridge_data(self, json_data: str):
        try:
            data = json.loads(json_data)
            self.status = data.get("status", "IDLE").upper()
            self.confidence = float(data.get("confidence", 0.0)) * 100
            self.risk_factors = data.get("risk_factors", [])
            
            if self.status == "SAFE":
                self.status_color = "green"
            elif self.status == "WARNING":
                self.status_color = "orange"
            elif self.status == "CRITICAL":
                self.status_color = "red"
            else:
                self.status_color = "gray"
        except Exception:
            pass

def video_panel():
    return rx.box(
        # 1. Header Row (Fixed Height)
        rx.hstack(
            rx.text("LIVE FEED - CAM 01", font_weight="bold", color="white"),
            rx.badge("LIVE", color_scheme="red", variant="solid"),
            justify="between",
            width="100%",
            padding_bottom="4"
        ),
        
        # 2. Video Wrapper (Flex Grow + Relative Positioning)
        # This box takes up ONLY the remaining space in the card.
        rx.box(
            rx.el.video(
                id="webcam-video",
                auto_play=True,
                plays_inline=True,
                muted=True,
                style={
                    "position": "absolute",  # <--- MAGIC FIX: Take video out of layout flow
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "object_fit": "contain", # Black bars if needed, but keeps aspect ratio
                    "background_color": "black"
                }
            ),
            width="100%",
            flex="1",              # Grow to fill remaining height
            position="relative",   # Anchor for the absolute video
            min_height="0",        # CSS trick to allow flex children to shrink
            border_radius="8px",
            overflow="hidden",     # Cut off anything sticking out
            bg="black"
        ),

        # 3. Main Container Styles
        id="video-container",
        display="flex",            # Turn card into a Flex Column
        flex_direction="column",
        height="100%",             # Force strict height match with Grid
        border="4px solid",
        border_color=State.status_color,
        padding="4",
        border_radius="xl",
        bg="gray.900",
        transition="border-color 0.2s ease"
    )

def status_panel():
    return rx.vstack(
        rx.card(
            rx.vstack(
                rx.heading(State.status, size="8", color=State.status_color),
                rx.text(f"Confidence: {State.confidence:.1f}%", size="4", color="gray.400"),
                align="center",
                spacing="2"
            ),
            width="100%",
            background_color="gray.800"
        ),
        rx.card(
            rx.vstack(
                rx.text("ACTIVE RISK FACTORS", font_weight="bold", font_size="xs", color="gray.500"),
                rx.cond(
                    State.risk_factors.length() > 0,
                    rx.foreach(State.risk_factors, lambda rf: rx.text(f"• {rf}", color="red.300", font_size="sm")),
                    rx.text("No ergonomic risks detected.", color="green.300", font_size="sm")
                ),
                align="start",
                spacing="2"
            ),
            width="100%",
            background_color="gray.800",
            flex="1"
        ),
        rx.card(
            rx.vstack(
                rx.button(
                    rx.cond(State.is_streaming, "STOP AI STREAM", "START AI STREAM"),
                    on_click=State.toggle_stream,
                    color_scheme=rx.cond(State.is_streaming, "red", "green"),
                    size="4",
                    width="100%"
                ),
                rx.text("Check browser console for [AI-STREAM] logs", font_size="10px", color="gray.600"),
            ),
            width="100%",
            background_color="gray.800"
        ),
        height="100%",
        spacing="4",
        width="100%"
    )

def index():
    return rx.box(
        # Load dashboard.js from root
        rx.script(src="/dashboard.js"),
        
        # Hidden Data Bridge
        rx.input(
            id="js-data-bridge",
            on_change=State.handle_bridge_data,
            style={"display": "none"}
        ),

        rx.hstack(
            rx.heading("Real-Time Industrial Posture Monitoring", size="5", color="white"),
            rx.text("Admin Dashboard", color="gray.500"),
            justify="between",
            padding="4",
            border_bottom="1px solid #333",
            bg="black"
        ),
        
        rx.grid(
            rx.box(video_panel(), grid_column="span 8"),
            rx.box(status_panel(), grid_column="span 4"),
            columns="12",
            gap="6",
            padding="6",
            height="90vh",
            bg="black"
        )
    )

app = rx.App(theme=rx.theme(appearance="dark"))
app.add_page(index)