"""A Gradio page that scores one booking. It uses only the classes."""

import gradio as gr

from noshows.appointment import (
    BOOKINGS,
    DISTANCE_LIMITS,
    INSURANCES,
    NUMBER_LIMITS,
    TREATMENTS,
    WEEKDAYS,
    Appointment,
)
from noshows.predictor import NoShowPredictor

try:  # present only on Hugging Face ZeroGPU hardware
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU
    def keep_zerogpu_happy() -> None:
        """Does nothing. ZeroGPU only needs one GPU function to start."""


predictor = NoShowPredictor()  # loads the model once, when the page starts


def check_booking(
    lead_days,
    age,
    prior_no_shows,
    hour,
    first_visit,
    distance_km,
    weekday,
    booking,
    insurance,
    treatment,
):
    """Build an Appointment from the page, score it, and describe the result."""
    try:
        appointment = Appointment(
            lead_days=lead_days,
            age=age,
            prior_no_shows=prior_no_shows,
            hour=hour,
            first_visit=int(first_visit),  # the Checkbox gives True or False
            distance_km=distance_km,
            weekday=weekday,
            booking=booking,
            insurance=insurance,
            treatment=treatment,
        )
    except ValueError as error:
        raise gr.Error(str(error)) from KeyError

    if appointment.distance_missing:
        gr.Warning("Distance was left empty, so the model used the typical distance.")

    prediction = predictor.predict(appointment)
    chance = f"{prediction.probability:.0%}"
    rule = f"Call when the chance is {predictor.threshold:.0%} or more."
    return chance, prediction.action, rule


def number_box(name):
    """A number field whose limits come from Appointment."""
    low, high = NUMBER_LIMITS[name]
    return gr.Number(label=name, info=f"{low} to {high}")


with gr.Blocks(title="Maple Physio no-show check") as demo:
    gr.Markdown("# Maple Physio no-show check\nEnter a booking to see the risk.")
    with gr.Row():
        with gr.Column():
            lead_days = number_box("lead_days")
            age = number_box("age")
            prior_no_shows = number_box("prior_no_shows")
            hour = number_box("hour")
            first_visit = gr.Checkbox(label="first_vist (tick if it is a first vist)")
            low, high = DISTANCE_LIMITS
            distance_km = gr.Number(
                label="distance_km", info=f"{low} to {high}, or leave empty"
            )
            weekday = gr.Dropdown(WEEKDAYS, value=WEEKDAYS[0], label="weekday")
            booking = gr.Dropdown(BOOKINGS, value=BOOKINGS[0], label="booking")
            insurance = gr.Dropdown(INSURANCES, value=INSURANCES[0], label="insurance")
            treatment = gr.Dropdown(TREATMENTS, value=TREATMENTS[0], label="treatment")
            button = gr.Button("Check booking", variant="primary")
        with gr.Column():
            chance_out = gr.Textbox(label="Chance of a no-show")
            action_out = gr.Textbox(label="What to do")
            rule_out = gr.Textbox(label="Rule used")
    button.click(
        check_booking,
        inputs=[
            lead_days,
            age,
            prior_no_shows,
            hour,
            first_visit,
            distance_km,
            weekday,
            booking,
            insurance,
            treatment,
        ],
        outputs=[chance_out, action_out, rule_out],
    )

if __name__ == "__main__":
    demo.launch()
