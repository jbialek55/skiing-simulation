"""Interfejs CLI dla symulacji stochastycznej F. Kamińskiego (2026)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from skiing_simulation.kaminski.model import SkiModel
from skiing_simulation.kaminski.visualizer import KaminskiVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skiing-kaminski-simulation",
        description="Skalowalny stochastyczny wieloagentowy model ruchu narciarzy na stoku (Kamiński 2026).",
    )
    parser.add_argument(
        "--skiers",
        type=int,
        default=16,
        help="Liczba narciarzy do zasymulowania (domyślnie: 16)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=40,
        help="Szerokość stoku w metrach (domyślnie: 40)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=350,
        help="Długość stoku w metrach (domyślnie: 350 - odcinek Kasprowego Wierchu)",
    )
    parser.add_argument(
        "--steps",
        "--max-steps",
        type=int,
        default=None,
        dest="max_steps",
        help="Opcjonalny limit kroków bezpieczeństwa (domyślnie: brak, symulacja działa aż wszyscy dojadą do końca)",
    )
    parser.add_argument(
        "--buffer-r",
        type=int,
        default=3,
        help="Szerokość strefy buforowej r przy krawędziach stoku (domyślnie: 3 m)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno generatora liczb losowych (domyślnie: 42)",
    )
    parser.add_argument(
        "--features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Włącz/wyłącz strefy trudności: płaty lodu i pole muld (domyślnie: włączone)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kaminski_simulation_result.png",
        help="Ścieżka do zapisu wykresów (domyślnie: data/kaminski_simulation_result.png)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="Opcjonalna ścieżka do zapisu animacji GIF (np. data/kaminski_demo.gif)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  🎿 Stochastyczny Model Narciarski (Kamiński AGH 2026)")
    print("=" * 70)
    print(f"Konfiguracja stoku: {args.width}m x {args.length}m, bufor r={args.buffer_r}m")
    limit_info = f"max_kroków={args.max_steps}" if args.max_steps is not None else "trwa do zjazdu wszystkich"
    print(f"Liczba agentów: {args.skiers}, seed={args.seed}, tryb: {limit_info}")

    features_info = "włączone (płaty lodu + pole muld)" if args.features else "wyłączone (jednolity stok)"
    print(f"Strefy trudności terenu: {features_info}")

    model = SkiModel(
        width=args.width,
        length=args.length,
        buffer_zone_r=args.buffer_r,
        use_kasprowy_profile=True,
        enable_terrain_features=args.features,
        seed=args.seed,
    )
    model.populate_skiers(count=args.skiers, stagger_y=True)

    print("\n▶ Trwa wykonywanie symulacji (aż wszyscy dojadą do końca)...")
    model.run_simulation(max_steps=args.max_steps)

    stats = model.get_summary_stats()
    print("\n" + "=" * 70)
    print("  Podsumowanie telemetrii:")
    print("=" * 70)
    print(f"• Ukończyło zjazd:         {stats['finished_skiers']} / {stats['total_skiers']}")
    print(f"• Całkowita liczba kroków: {stats['total_steps']}")
    print(f"• Wykryte kolizje:         {stats['total_collisions']}")
    print(f"• Średnia prędkość:        {stats['average_speed_ms']:.2f} m/s ({stats['average_speed_kmh']:.1f} km/h)")
    print(f"• Maksymalna prędkość:     {stats['max_speed_kmh']:.1f} km/h")
    print(f"• Zarejestrowane zakręty:  {stats['total_turns_recorded']}")
    if args.features:
        print(f"• Czas w strefie lodu:     {stats['total_time_ice']} kroków")
        print(f"• Czas w polu muld:        {stats['total_time_moguls']} kroków")
        print(f"• Poślizgi na lodzie:      {stats['total_ice_slips']}")

    visualizer = KaminskiVisualizer(model)
    saved_path = visualizer.render_dashboard(output_path=args.output)
    print(f"\n📊 Wygenerowano raport graficzny: {saved_path}")

    if args.gif:
        print(f"🎬 Generowanie animacji GIF do: {args.gif}...")
        gif_path = visualizer.render_animation(output_path=args.gif, fps=8)
        print(f"✅ Animacja zapisana: {gif_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
