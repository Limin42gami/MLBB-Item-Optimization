import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_CANDIDATES = (
    PROJECT_ROOT / "Data",
    PROJECT_ROOT / "MLBB-API-main (Database)" / "v1",
)


HERO_TO_EMBLEM = {
    "Marksman": "Marksman",
    "Assassin": "Assassin",
    "Mage": "Mage",
    "Fighter": "Fighter",
    "Tank": "Tank",
    "Support": "Support",
}

CLASS_TO_DAMAGE_TYPE = {
    "Marksman": "physical",
    "Assassin": "physical",
    "Fighter": "physical",
    "Tank": "mixed",
    "Support": "mixed",
    "Mage": "magic",
}

CATEGORY_AFFINITY = {
    "Marksman": {"Attack": 1.0, "Defense": 0.35, "Magic": 0.0, "Jungle": 0.6, "Roam": 0.0},
    "Assassin": {"Attack": 0.95, "Defense": 0.3, "Magic": 0.0, "Jungle": 1.0, "Roam": 0.0},
    "Mage": {"Attack": 0.0, "Defense": 0.35, "Magic": 1.0, "Jungle": 0.45, "Roam": 0.0},
    "Fighter": {"Attack": 0.75, "Defense": 0.8, "Magic": 0.0, "Jungle": 0.6, "Roam": 0.0},
    "Tank": {"Attack": 0.15, "Defense": 1.0, "Magic": 0.2, "Jungle": 0.2, "Roam": 0.85},
    "Support": {"Attack": 0.1, "Defense": 0.75, "Magic": 0.55, "Jungle": 0.1, "Roam": 1.0},
}

ANTI_MAGIC_KEYWORDS = ("athena", "radiant", "oracle")
ANTI_PHYSICAL_KEYWORDS = ("blade armor", "antique cuirass", "brute force", "dominance")
ANTI_HEAL_KEYWORDS = ("dominance", "necklace of durance", "sea halberd")
PENETRATION_KEYWORDS = ("malefic", "divine glaive", "genius wand")


@dataclass
class MatchContext:
    hero_name: str
    enemy_names: List[str]
    ally_names: List[str]
    game_duration_min: int
    rank: str
    skill_level: str
    damage_focus: str


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_db_root(data_dir: Path = None) -> Path:
    if data_dir is not None:
        candidate_dirs = (Path(data_dir),)
    else:
        candidate_dirs = DEFAULT_DB_CANDIDATES

    required = ("hero-meta-final.json", "item-meta-final.json", "emblem-meta-final.json")
    for directory in candidate_dirs:
        if all((directory / name).exists() for name in required):
            return directory

    searched_paths = ", ".join(str(p) for p in candidate_dirs)
    raise FileNotFoundError(
        "Could not locate MLBB data files. Expected "
        f"{', '.join(required)} in one of: {searched_paths}"
    )


def normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def parse_numeric(value) -> float:
    if value is None:
        return 0.0
    raw = str(value).replace("%", "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def extract_item_meta(item: Dict) -> Dict:
    item_data = (item.get("data") or [{}])[0]
    modifiers_list = item_data.get("modifiers") or []
    modifiers = modifiers_list[0] if modifiers_list else {}
    return {
        "name": item.get("item_name", ""),
        "id": item.get("id", ""),
        "category": item.get("item_category", "Unknown"),
        "tier": int(parse_numeric(item.get("item_tier", 0))),
        "cost": int(parse_numeric(item_data.get("cost", 0))),
        "summary": str(item_data.get("summary", "")).lower(),
        "modifiers": modifiers,
    }


def build_hero_index(hero_data: List[Dict]) -> Dict[str, Dict]:
    return {normalize_name(hero.get("hero_name", "")): hero for hero in hero_data}


def infer_damage_profile(hero_names: List[str], hero_index: Dict[str, Dict]) -> Dict[str, int]:
    profile = {"physical": 0, "magic": 0, "mixed": 0}
    for name in hero_names:
        hero = hero_index.get(normalize_name(name))
        if not hero:
            continue
        dmg = CLASS_TO_DAMAGE_TYPE.get(hero.get("class", ""), "mixed")
        profile[dmg] += 1
    return profile


def count_classes(hero_names: List[str], hero_index: Dict[str, Dict]) -> Dict[str, int]:
    classes = {k: 0 for k in CATEGORY_AFFINITY}
    for name in hero_names:
        hero = hero_index.get(normalize_name(name))
        if not hero:
            continue
        role = hero.get("class", "")
        if role in classes:
            classes[role] += 1
    return classes


def one_hot_features(context: MatchContext, hero_index: Dict[str, Dict], all_hero_classes: List[str]) -> Tuple[List[str], List[int]]:
    hero = hero_index.get(normalize_name(context.hero_name), {})
    hero_class = hero.get("class", "Unknown")
    enemy_class_counts = count_classes(context.enemy_names, hero_index)
    ally_class_counts = count_classes(context.ally_names, hero_index)

    feature_names = []
    feature_values = []

    for cls in all_hero_classes:
        feature_names.append(f"hero_class__{cls}")
        feature_values.append(1 if hero_class == cls else 0)

    for cls in all_hero_classes:
        feature_names.append(f"enemy_count__{cls}")
        feature_values.append(enemy_class_counts.get(cls, 0))

    for cls in all_hero_classes:
        feature_names.append(f"ally_count__{cls}")
        feature_values.append(ally_class_counts.get(cls, 0))

    for rank in ("warrior", "elite", "master", "grandmaster", "epic", "legend", "mythic", "mythical honor", "mythical glory"):
        feature_names.append(f"rank__{rank}")
        feature_values.append(1 if normalize_name(context.rank) == rank else 0)

    for skill in ("beginner", "intermediate", "advanced", "pro"):
        feature_names.append(f"skill__{skill}")
        feature_values.append(1 if normalize_name(context.skill_level) == skill else 0)

    for dmg in ("balanced", "physical", "magic", "durability"):
        feature_names.append(f"damage_focus__{dmg}")
        feature_values.append(1 if normalize_name(context.damage_focus) == dmg else 0)

    feature_names.append("game_duration_min")
    feature_values.append(max(0, context.game_duration_min))
    return feature_names, feature_values


def suggest_emblem(context: MatchContext, hero: Dict, emblems: List[Dict], hero_index: Dict[str, Dict]) -> Dict:
    default_emblem = HERO_TO_EMBLEM.get(hero.get("class", ""), "Common")
    enemy_profile = infer_damage_profile(context.enemy_names, hero_index)
    enemy_magic_heavy = enemy_profile["magic"] >= 3
    enemy_physical_heavy = enemy_profile["physical"] >= 3

    emblem_by_name = {normalize_name(e.get("emblem_name", "")): e for e in emblems}
    pick = default_emblem
    if hero.get("class") in ("Tank", "Support") and enemy_magic_heavy:
        pick = "Tank"
    elif hero.get("class") == "Mage" and enemy_physical_heavy:
        pick = "Mage"

    return emblem_by_name.get(normalize_name(pick), emblems[0] if emblems else {"emblem_name": "Common"})


def hero_matchup_signal(hero: Dict, context: MatchContext) -> float:
    counters = {normalize_name(c.get("heroname", "")) for c in hero.get("counters", [])}
    synergies = {normalize_name(s.get("heroname", "")) for s in hero.get("synergies", [])}
    enemy_hits = sum(1 for e in context.enemy_names if normalize_name(e) in counters)
    ally_hits = sum(1 for a in context.ally_names if normalize_name(a) in synergies)
    return (enemy_hits * 0.12) + (ally_hits * 0.08)


def score_item(item: Dict, hero_class: str, context: MatchContext, hero_index: Dict[str, Dict]) -> float:
    name = normalize_name(item["name"])
    category = item["category"]
    cost = item["cost"]
    tier = item["tier"]

    base = CATEGORY_AFFINITY.get(hero_class, {}).get(category, 0.2) * 2.2
    early_bonus = 0.0
    if context.game_duration_min <= 8:
        early_bonus += 0.8 if (tier <= 2 or cost <= 1500) else -0.2
    elif context.game_duration_min >= 16:
        early_bonus += 0.35 if tier >= 3 else -0.15

    rank = normalize_name(context.rank)
    skill = normalize_name(context.skill_level)
    rank_bonus = 0.1 if rank in ("mythic", "mythical honor", "mythical glory") else 0.0
    skill_bonus = 0.1 if skill in ("advanced", "pro") and tier >= 3 else 0.0

    enemy_classes = count_classes(context.enemy_names, hero_index)
    enemy_profile = infer_damage_profile(context.enemy_names, hero_index)
    anti_magic = 0.5 if enemy_profile["magic"] >= 3 and any(k in name for k in ANTI_MAGIC_KEYWORDS) else 0.0
    anti_physical = 0.5 if enemy_profile["physical"] >= 3 and any(k in name for k in ANTI_PHYSICAL_KEYWORDS) else 0.0
    anti_heal = 0.4 if (enemy_classes["Fighter"] + enemy_classes["Tank"] + enemy_classes["Support"] >= 3) and any(k in name for k in ANTI_HEAL_KEYWORDS) else 0.0
    anti_tank = 0.35 if enemy_classes["Tank"] >= 2 and any(k in name for k in PENETRATION_KEYWORDS) else 0.0

    dmg_focus = normalize_name(context.damage_focus)
    focus_bonus = 0.0
    if dmg_focus == "physical" and category == "Attack":
        focus_bonus += 0.45
    elif dmg_focus == "magic" and category == "Magic":
        focus_bonus += 0.45
    elif dmg_focus == "durability" and category == "Defense":
        focus_bonus += 0.45
    elif dmg_focus == "balanced":
        focus_bonus += 0.15

    return base + early_bonus + rank_bonus + skill_bonus + anti_magic + anti_physical + anti_heal + anti_tank + focus_bonus


def suggest_items(context: MatchContext, hero: Dict, raw_items: List[Dict], hero_index: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    hero_class = hero.get("class", "Unknown")
    items = [extract_item_meta(item) for item in raw_items]
    scored = []
    for item in items:
        if item["cost"] <= 0:
            continue
        score = score_item(item, hero_class, context, hero_index)
        scored.append({**item, "score": score})

    scored.sort(key=lambda i: (i["score"], -i["cost"]), reverse=True)
    core = scored[:3]
    situational = [i for i in scored if i["id"] not in {c["id"] for c in core}][:3]
    return {"core": core, "situational": situational}


def build_order(core_items: List[Dict], context: MatchContext) -> List[Dict]:
    if context.game_duration_min <= 8:
        return sorted(core_items, key=lambda item: (item["cost"], -item["score"]))
    return sorted(core_items, key=lambda item: (-item["score"], item["cost"]))


def build_path_priority(items: List[Dict], all_items: List[Dict]) -> Dict[str, List[Dict]]:
    by_category = {}
    for item in all_items:
        by_category.setdefault(item["category"], []).append(item)
    for category_items in by_category.values():
        category_items.sort(key=lambda x: (x["tier"], x["cost"]))

    output = {}
    for core in items:
        category_pool = by_category.get(core["category"], [])
        starter = [i for i in category_pool if i["tier"] <= 2 and i["cost"] < core["cost"] and i["id"] != core["id"]][:2]
        output[core["name"]] = starter
    return output


def choose_hero(prompt: str, all_names: List[str], default_multi: bool = False) -> List[str]:
    print(f"\n{prompt}")
    print("Type names separated by commas, or press Enter to skip.")
    typed = input("> ").strip()
    if not typed and default_multi:
        return []

    picks = [name.strip() for name in typed.split(",") if name.strip()]
    validated = []
    name_map = {normalize_name(n): n for n in all_names}
    for raw in picks:
        match = name_map.get(normalize_name(raw))
        if match:
            validated.append(match)
        else:
            print(f"  - '{raw}' not found and will be skipped.")
    return validated


def get_context(hero_names: List[str]) -> MatchContext:
    hero_name = choose_hero("1) Pick your hero", hero_names)[0]
    enemies = choose_hero("2) Pick enemy heroes", hero_names, default_multi=True)
    allies = choose_hero("3) Pick ally heroes (optional)", hero_names, default_multi=True)

    game_duration = int(input("\n4) Game duration (minutes): ").strip() or "10")
    rank = input("5) Rank (e.g. Epic/Mythic): ").strip() or "Epic"
    skill = input("6) Skill level (Beginner/Intermediate/Advanced/Pro): ").strip() or "Intermediate"
    damage_focus = input("7) Damage focus (Balanced/Physical/Magic/Durability): ").strip() or "Balanced"

    return MatchContext(
        hero_name=hero_name,
        enemy_names=enemies,
        ally_names=allies,
        game_duration_min=game_duration,
        rank=rank,
        skill_level=skill,
        damage_focus=damage_focus,
    )


def print_result(hero: Dict, emblem: Dict, items: Dict[str, List[Dict]], order: List[Dict], path: Dict[str, List[Dict]], matchup_score: float) -> None:
    print("\n=== MLBB Item Optimization v1 ===")
    print(f"Hero: {hero.get('hero_name', 'Unknown')} ({hero.get('class', 'Unknown')})")
    print(f"Suggested Emblem: {emblem.get('emblem_name', 'Common')}")
    print(f"Counter/Synergy Signal: {matchup_score:.2f}")

    print("\nCore Items:")
    for idx, item in enumerate(items["core"], start=1):
        print(f"{idx}. {item['name']} - score {item['score']:.2f} (cost {item['cost']})")

    print("\nSituational Items:")
    for idx, item in enumerate(items["situational"], start=1):
        print(f"{idx}. {item['name']} - score {item['score']:.2f} (cost {item['cost']})")

    print("\nBuild Order Priority:")
    for idx, item in enumerate(order, start=1):
        print(f"{idx}. Finish {item['name']}")

    print("\nBasic Item Priority Before Core Completion:")
    for core_name, starters in path.items():
        if not starters:
            print(f"- {core_name}: no explicit basic path found; rush this core when ahead.")
            continue
        parts = ", ".join(f"{i['name']} (cost {i['cost']})" for i in starters)
        print(f"- {core_name}: {parts}")


def main() -> None:
    db_root = resolve_db_root()
    hero_payload = read_json(db_root / "hero-meta-final.json")
    item_payload = read_json(db_root / "item-meta-final.json")
    emblem_payload = read_json(db_root / "emblem-meta-final.json")

    heroes = hero_payload.get("data", [])
    items_raw = item_payload.get("data", [])
    emblems = emblem_payload.get("data", [])

    hero_index = build_hero_index(heroes)
    hero_names = sorted(hero.get("hero_name", "") for hero in heroes if hero.get("hero_name") and hero.get("hero_name") != "None")
    all_classes = sorted(k for k in CATEGORY_AFFINITY.keys())

    context = get_context(hero_names)
    hero = hero_index.get(normalize_name(context.hero_name))
    if not hero:
        raise ValueError("Selected hero is invalid.")

    feature_names, feature_values = one_hot_features(context, hero_index, all_classes)
    print(f"\nFeature engineering complete: {len(feature_names)} one-hot/numeric features generated.")

    matchup_score = hero_matchup_signal(hero, context)
    emblem = suggest_emblem(context, hero, emblems, hero_index)
    item_recs = suggest_items(context, hero, items_raw, hero_index)
    full_item_meta = [extract_item_meta(i) for i in items_raw]
    order = build_order(item_recs["core"], context)
    path = build_path_priority(order, full_item_meta)
    print_result(hero, emblem, item_recs, order, path, matchup_score)

    print("\nModeling roadmap:")
    print("- Classification: Logistic Regression / Random Forest / XGBoost on one-hot features")
    print("- Ranking: LightGBM Ranker with pairwise preference labels")
    print("- Sequence: LSTM/Transformer for next-item prediction by timestamped matches")


if __name__ == "__main__":
    main()
