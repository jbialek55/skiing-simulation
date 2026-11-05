import argparse
import os
import numpy as np
from skiing_simulation.slope.SlopeModel import SkiModel
from skiing_simulation.visualize.visualizer import SlopeVisualizer
from skiing_simulation.agents.base_skier import BaseSkier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symulacja zjazdu narciarskiego wg Holleczek & Tröster (2012) Phys. Rev. E 85, 056101"
    )
    parser.add_argument("--slope", type=str, default="jochstock", choices=["jochstock", "graustock"],
                        help="Wybór stoku ze szwajcarskiego Engelberg z Table I: 'jochstock' (889m, intermediate) lub 'graustock' (1250m, expert)")
    parser.add_argument("--skiers", type=int, default=1, help="Liczba narciarzy do wygenerowania na starcie (domyślnie 1)")
    parser.add_argument("--dt", type=float, default=0.01, help="Krok integracji w sekundach (domyślnie 0.01s = 10ms, Sec. IV.A)")
    parser.add_argument("--mu", type=float, default=0.10, help="Współczynnik tarcia kinetycznego nart (domyślnie 0.10, Table IV)")
    parser.add_argument("--rsc", type=float, default=10.0, help="Promień taliowania nart R_sc w metrach (domyślnie 10m, Table IV)")
    parser.add_argument("--delta", type=float, default=10.0, help="Próg odchylenia kierunku delta w stopniach (domyślnie 10 deg, Table II)")
    parser.add_argument("--speedup", type=int, default=6, help="Przyspieszenie animacji: liczba kroków fizyki na klatkę (domyślnie 6x, np. 10 lub 15)")
    parser.add_argument("--animate", action="store_true", help="Uruchom interaktywne okno animacji")
    parser.add_argument("--gif", type=str, default=None, help="Ścieżka do zapisu animacji GIF (np. data/simulation.gif)")
    parser.add_argument("--output", type=str, default="data/simulation_result.png", help="Ścieżka do zapisu wykresu podsumowania")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 75)
    print("  MODEL TRAFIKU NARCIARSKIEGO (Particle-based model for skiing traffic)")
    print("  Thomas Holleczek & Gerhard Tröster (ETH Zurich)")
    print("  Physical Review E 85, 056101 (2012)")
    print("=" * 75)

    model = SkiModel(
        slope_name=args.slope,
        dt=args.dt,
    )

    ref_sim_speed = 55.2 if args.slope == "jochstock" else 54.1
    ref_exp_speed = 51.8 if args.slope == "jochstock" else 42.1

    print(f"Stok: {args.slope.upper()} (Engelberg, Szwajcaria)")
    print(f"  - Długość:              {model.height:.0f} m")
    print(f"  - Średnia szerokość:    {model.width:.0f} m")
    print(f"  - Wysokość start/meta:  {model.terrain.start_altitude:.0f} m / {model.terrain.finish_altitude:.0f} m (spadek: {model.terrain.drop:.0f} m)")
    print(f"  - Średnie nachylenie:   {model.terrain.avg_inclination * 100:.1f}%")
    print(f"  - Krok całkowania dt:   {args.dt * 1000:.0f} ms (Sec. IV.A)")
    print(f"  - Tarcie nart mu:       {args.mu:.2f} (Table IV)")
    print(f"  - Promień taliowania R: {args.rsc:.1f} m (Table IV)")
    print("-" * 75)

    print(f"Dodawanie {args.skiers} narciarzy na szczycie stoku...")
    left_b, right_b = model.terrain.get_borders_at(5.0)
    w_start = right_b - left_b
    for i in range(args.skiers):
        if args.skiers == 1:
            x_start = (left_b + right_b) / 2.0
        else:
            x_start = left_b + 6.0 + i * ((w_start - 12.0) / max(args.skiers - 1, 1))
        model.add_skier(
            x=x_start,
            y=5.0,
            mass=85.0,
            sidecut_radius=args.rsc,
            friction_coeff=args.mu,
            threshold_delta_deg=args.delta,
        )

    viz = SlopeVisualizer(model)

    if args.animate or args.gif:
        print("Uruchamianie trybu animacji...")
        viz.animate(max_steps=15000, interval=20, steps_per_frame=args.speedup, save_path=args.gif)
    else:
        print("Symulacja zjazdu narciarzy...")
        max_steps = 15000
        for step_num in range(max_steps):
            model.step()
            stats = model.get_summary_stats()
            if stats["active"] == 0:
                print(f"Wszyscy narciarze ukończyli zjazd po {step_num} krokach integracji ({stats['time_s']:.2f} s)!")
                break

    # Podsumowanie i statystyki wyświetlane ZAWSZE po zakończeniu (zarówno po animacji, jak i bez)
    stats = model.get_summary_stats()
    all_skiers = [a for a in model.agents if isinstance(a, BaseSkier)]
    all_speeds = []

    print("\n" + "=" * 75)
    print("  PODSUMOWANIE I STATYSTYKI ZJAZDU (Holleczek & Tröster 2012)")
    print("=" * 75)

    for skier in all_skiers:
        if not skier.trajectory:
            continue
        traj = np.array([(p[0], p[1], p[2] * 3.6, p[4]) for p in skier.trajectory])
        speeds = traj[:, 2]
        all_speeds.extend(speeds)

        diff_x = np.diff(traj[:, 0])
        apex_count = len(np.where(np.diff(np.sign(diff_x)))[0]) if len(diff_x) > 1 else 0
        turns_per_100m = (apex_count / (model.height / 100.0))

        status_label = "UKOŃCZONO" if skier.status == "finished" else skier.status.upper()
        skier_time = len(traj) * model.dt
        print(f"\n--- NARCIARZ #{skier.skier_id} [{status_label}] ---")
        print(f"  - Czas zjazdu:          {skier_time:.2f} s")
        print(f"  - Średnia prędkość:     {speeds.mean():.2f} km/h (Artykuł: exp={ref_exp_speed} km/h, sim={ref_sim_speed} km/h)")
        print(f"  - Mediana prędkości p50:{np.median(speeds):.2f} km/h (Artykuł Fig. 12: ~54 km/h)")
        print(f"  - Kwartyle p25 / p75:   {np.percentile(speeds, 25):.2f} km/h / {np.percentile(speeds, 75):.2f} km/h")
        print(f"  - Min / Max prędkość:   {speeds.min():.2f} km/h / {speeds.max():.2f} km/h")
        print(f"  - Liczba skrętów:       {apex_count} ({turns_per_100m:.2f} skrętów/100m, Artykuł: 2.0 - 3.6)")
        print(f"  - Zakres poprzeczny X:  [{traj[:, 0].min():.1f}m, {traj[:, 0].max():.1f}m] (szerokość korytarza: {traj[:, 0].max() - traj[:, 0].min():.1f}m)")

    if len(all_skiers) > 1 and all_speeds:
        all_speeds_arr = np.array(all_speeds)
        print("\n" + "-" * 75)
        print(f"PODSUMOWANIE DLA CAŁEJ GRUPY ({len(all_skiers)} narciarzy):")
        print(f"  - Średnia prędkość grupy: {all_speeds_arr.mean():.2f} km/h")
        print(f"  - Mediana prędkości grupy:{np.median(all_speeds_arr):.2f} km/h")
        print(f"  - Łączna liczba ukończonych: {stats['finished']}/{len(all_skiers)}")
    print("=" * 75)

    print(f"\nZapisywanie wykresu podsumowującego do '{args.output}'...")
    viz.render_static_result(args.output)
    print("Podsumowanie zapisane pomyślnie!")


if __name__ == "__main__":
    main()
