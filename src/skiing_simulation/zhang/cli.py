"""CLI dla modelu probabilistycznego Zhang i in. (2024)."""

from __future__ import annotations

import argparse

from skiing_simulation.zhang.model import ZhangModel
from skiing_simulation.zhang.visualizer import ZhangVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automat komórkowy Zhang i in. (2024)")
    parser.add_argument("--width", type=int, default=450, help="Szerokość stoku w komórkach")
    parser.add_argument("--length", type=int, default=750, help="Długość stoku w komórkach")
    parser.add_argument("--skiers", type=int, default=1, help="Liczba niezależnych narciarzy")
    parser.add_argument("--delta", type=float, default=0.012, help="Współczynnik zmiany prawdopodobieństwa δ")
    parser.add_argument("--epsilon", type=float, default=0.87, help="Próg przełączenia strony skrętu ε")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno generatora losowego")
    parser.add_argument("--initial-direction", choices=["left", "right", "random"], default="random")
    parser.add_argument(
        "--start-margin", type=int, default=125,
        help="Minimalny odstęp losowanego startu od krawędzi [komórki] (domyślnie 125)",
    )
    parser.add_argument("--animate", action="store_true", help="Otwiera animację Matplotlib")
    parser.add_argument("--gif", type=str, default=None, help="Zapisuje animację GIF pod podaną ścieżką")
    parser.add_argument("--steps-per-frame", type=int, default=3, help="Kroki CA wykonywane na jedną klatkę")
    parser.add_argument("--max-frames", type=int, default=1000, help="Limit klatek animacji")
    parser.add_argument("--output", default="data/zhang_simulation_result.png", help="Plik PNG podsumowania")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skiers < 1:
        raise SystemExit("--skiers musi być dodatnie")
    if args.start_margin < 0 or 2 * args.start_margin >= args.width:
        raise SystemExit("--start-margin musi być nieujemny i pozostawiać co najmniej jedną komórkę startową")
    model = ZhangModel(
        width=args.width, length=args.length, delta=args.delta, epsilon=args.epsilon, seed=args.seed,
    )
    direction = None if args.initial_direction == "random" else args.initial_direction
    for _ in range(args.skiers):
        x = int(model.rng.integers(args.start_margin, args.width - args.start_margin))
        model.add_skier(x=x, direction=direction)
    visualizer = ZhangVisualizer(model)

    if args.animate or args.gif:
        visualizer.animate(args.gif, steps_per_frame=args.steps_per_frame, max_frames=args.max_frames)
    else:
        model.run()
    visualizer.render_static_result(args.output)
    summary = model.summary()
    print(
        f"Zhang et al. (2024): ukończyli {summary['finished']}/{summary['skiers']}; "
        f"kroki={summary['steps']}; udział ruchów bocznych={summary['lateral_share']:.1%}."
    )
