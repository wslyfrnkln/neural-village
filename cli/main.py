from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """neural-playback — run your music through a brain."""
    pass


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Path to audio file (MP3, WAV, FLAC, M4A)")
@click.option("--output", "-o", "output_dir", default="./output", show_default=True, help="Output directory")
@click.option("--device", default=None, help="Compute device override: cuda, mps, cpu")
@click.option("--track-name", "-n", "track_name", default=None, help="Track name for report card header")
@click.option("--no-render", is_flag=True, default=False, help="Skip brain render (faster)")
@click.option("--no-chart", is_flag=True, default=False, help="Skip temporal chart")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "both"]), default="both", show_default=True, help="Report card output format")
def analyze(input_path, output_dir, device, track_name, no_render, no_chart, output_format):
    """Analyze an audio file and output brain activation renders + report card."""
    from neural_playback.preprocess import preprocess_audio
    from neural_playback.inference import load_model, predict_brain_response
    from neural_playback.roi_mapping import load_destrieux_labels, load_roi_map, get_roi_timeseries
    from neural_playback.report_card import generate_report_card
    from neural_playback.visualization import render_brain_annotated, create_temporal_chart

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if track_name is None:
        track_name = input_path.stem

    click.echo(f"neural-playback: {input_path.name}")

    # Preprocess
    click.echo("Preprocessing audio...")
    mp4_path = preprocess_audio(input_path, output_dir=output_dir / "tmp")

    # Load model + run inference
    click.echo("Loading model...")
    model = load_model(device=device)
    click.echo("Running brain prediction...")
    preds, segments = predict_brain_response(model, mp4_path)
    click.echo(f"Output shape: {preds.shape}")

    # Load atlas
    labels = load_destrieux_labels()
    roi_map = load_roi_map()

    saved = []

    # Brain render
    if not no_render:
        click.echo("Rendering annotated brain...")
        render_path = output_dir / "brain_activation.png"
        render_brain_annotated(preds, roi_map, labels, timestep=None, threshold=0.5,
                                output_path=render_path, track_name=track_name)
        saved.append(str(render_path))

    # Temporal chart
    if not no_chart:
        click.echo("Building temporal chart...")
        timeseries = get_roi_timeseries(preds, labels, roi_map)
        chart_path = output_dir / "timeline.html"
        create_temporal_chart(timeseries, title=f"Neural Activation — {track_name}",
                               output_path=chart_path)
        saved.append(str(chart_path))

    # Report card
    scores, report = generate_report_card(preds, labels, track_name=track_name, roi_map=roi_map)

    if output_format in ("text", "both"):
        txt_path = output_dir / "report_card.txt"
        txt_path.write_text(report)
        saved.append(str(txt_path))

    if output_format in ("json", "both"):
        json_path = output_dir / "report_card.json"
        json_path.write_text(json.dumps({"track": str(input_path), "scores": scores}, indent=2))
        saved.append(str(json_path))

    click.echo("\n" + report)
    click.echo(f"\nSaved {len(saved)} files to {output_dir}/")
    for f in saved:
        click.echo(f"  {f}")


if __name__ == "__main__":
    cli()
