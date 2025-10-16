# llm_parser.py — LLM weather/disease parser (expanded with more weather constraints)

from typing import Dict, List

def _k(s: List[str], out_key: str, mult: float, msg: str):
    return {"syn": s, "key": out_key, "mult": mult, "msg": msg}

RULES = [
    _k(["extreme heat","severe heat","record heat"], "irrigate_multiplier", 1.35, "Extreme heat — irrigate more, prevent scorch."),
    _k(["heatwave","hot spell","high temperature","heat index"], "irrigate_multiplier", 1.20, "Heatwave — increase irrigation."),
    _k(["drought","dry spell","water stress","prolonged dryness"], "irrigate_multiplier", 1.40, "Drought — irrigate more; schedule water smartly."),
    _k(["low humidity","very dry air"], "irrigate_multiplier", 1.15, "Low humidity — irrigate slightly more."),
    _k(["high uv","strong uv","uv index high","solar radiation high"], "monitor_multiplier", 1.15, "High UV — monitor for leaf scorch."),

    _k(["cold snap","cold wave","unseasonal cold"], "fertilize_multiplier", 0.9, "Cold snap — ease fertilization; protect seedlings."),
    _k(["frost","ground frost","frost risk"], "monitor_multiplier", 1.25, "Frost risk — monitor sensitive plots, consider covers."),
    _k(["wind chill"], "monitor_multiplier", 1.15, "Wind chill — inspect for cold damage."),

    _k(["high humidity","humid conditions","mugginess","sticky air"], "fungicide_multiplier", 1.20, "High humidity — apply fungicide when needed."),
    _k(["fog","dense fog","misty"], "fungicide_multiplier", 1.15, "Fog — fungus risk up; apply fungicide if symptoms appear."),
    _k(["dew","heavy dew"], "fungicide_multiplier", 1.10, "Heavy dew — watch mildew; treat promptly."),

    _k(["light rain","drizzle"], "irrigate_multiplier", 0.9, "Light rain — slightly reduce irrigation."),
    _k(["moderate rain"], "irrigate_multiplier", 0.8, "Moderate rain — reduce irrigation."),
    _k(["heavy rain","downpour","monsoon burst"], "irrigate_multiplier", 0.6, "Heavy rain — cut irrigation; check drainage."),
    _k(["very heavy rain","extreme rain"], "irrigate_multiplier", 0.5, "Extreme rain — stop irrigation; drain excess water."),
    _k(["flood","waterlogging","standing water"], "monitor_multiplier", 1.2, "Flooding — monitor drainage; avoid over-watering."),
    _k(["monsoon onset","monsoon active"], "irrigate_multiplier", 0.8, "Monsoon active — reduce irrigation plan."),
    _k(["monsoon break","break in monsoon"], "irrigate_multiplier", 1.15, "Monsoon break — irrigate more than usual."),

    _k(["strong wind","gusty","storm winds"], "monitor_multiplier", 1.15, "High winds — check lodging and damage."),
    _k(["dust storm","dusty winds"], "pesticide_multiplier", 1.10, "Dust storm — pests spread faster; be ready to treat."),
    _k(["thunderstorm","lightning"], "monitor_multiplier", 1.20, "Thunderstorm — inspect damage; delay risky ops."),
    _k(["hail","hailstorm"], "monitor_multiplier", 1.25, "Hail — inspect damage immediately."),
    _k(["cyclone","typhoon","hurricane"], "monitor_multiplier", 1.3, "Cyclone — expect major damage; survey fields carefully."),

    _k(["aphid","aphids"], "pesticide_multiplier", 1.35, "Pests up — apply pesticide."),
    _k(["locust","locusts"], "pesticide_multiplier", 1.4, "Locust alert — apply pesticide aggressively."),
    _k(["armyworm","cutworm"], "pesticide_multiplier", 1.3, "Armyworm — apply pesticide promptly."),
    _k(["mite","mites"], "pesticide_multiplier", 1.2, "Mites — apply pesticide where detected."),
    _k(["beetle","weevil"], "pesticide_multiplier", 1.2, "Beetle activity — apply pesticide as needed."),

    _k(["blight","late blight","early blight"], "fungicide_multiplier", 1.35, "Blight risk — apply fungicide."),
    _k(["rust"], "fungicide_multiplier", 1.25, "Rust risk — apply fungicide on affected rows."),
    _k(["mildew","powdery mildew","downy mildew"], "fungicide_multiplier", 1.3, "Mildew — apply fungicide."),
    _k(["wilt","bacterial wilt","fusarium"], "monitor_multiplier", 1.2, "Wilt symptoms — monitor and treat if needed."),

    _k(["low nitrogen","n deficiency","nitrogen deficit"], "fertilize_multiplier", 1.3, "N low — fertilize."),
    _k(["low phosphate","p deficiency"], "fertilize_multiplier", 1.2, "P low — fertilize."),
    _k(["low potassium","k deficiency"], "fertilize_multiplier", 1.2, "K low — fertilize."),
    _k(["salinity","saline soil"], "irrigate_multiplier", 1.1, "Salinity — flush with irrigation if feasible."),
    _k(["acidic soil","low ph"], "fertilize_multiplier", 1.05, "Acidic soil — adjust nutrients."),
    _k(["alkaline soil","high ph"], "fertilize_multiplier", 1.05, "Alkaline soil — adjust nutrients."),

    _k(["increase monitor","intense scouting","survey","monitoring drive"], "monitor_multiplier", 1.2, "Increase monitoring — map hotspots and act fast.")
]

BASELINE = {
    "irrigate_multiplier": 1.0,
    "pesticide_multiplier": 1.0,
    "fungicide_multiplier": 1.0,
    "monitor_multiplier": 1.0,
    "fertilize_multiplier": 1.0,
}

def parse_report(report_text: str) -> Dict:
    text = (report_text or "").lower()
    multipliers = dict(BASELINE)
    suggestions = []
    for r in RULES:
        if any(tok in text for tok in r["syn"]):
            k=r["key"]; curr=multipliers.get(k,1.0)
            if r["mult"]>=1.0: multipliers[k]=max(curr,r["mult"])
            else: multipliers[k]=min(curr,r["mult"])
            suggestions.append(r["msg"])
    if not suggestions:
        suggestions.append("No special alerts — follow standard best practices.")
    seen=set(); brief=[]
    for s in suggestions:
        if s not in seen:
            brief.append(s); seen.add(s)
    return {"multipliers": multipliers, "summary": brief}
