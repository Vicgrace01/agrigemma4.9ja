import csv
import json
import os

MASTER_DATA = [
    {
        "keywords": "swine pig pork livestock animal breeding feed disease african swine fever asf biosecurity mortality isolation disinfection swill quarantine symptoms",
        "content": "Swine (pig) production requires strict biosecurity, clean housing, and balanced energy-protein feed. African Swine Fever (ASF) is a highly contagious viral hemorrhagic disease with near 100% mortality. Clinical signs include high fever, skin cyanosis, and bloody diarrhea. No vaccine exists; enforce strict biosecurity, ban unboiled swill, and immediately quarantine or depopulate infected herds."
    },
    {
        "keywords": "corn maize grain cereal planting spacing fertilizer NPK armyworm pest control savanna silking tassel",
        "content": "Maize (corn) is a primary cereal crop across Nigerian savannas. Plant at 75cm x 25cm spacing during early rains. Apply NPK fertilizer via split application at 3 and 6 weeks post-planting. Monitor for Fall Armyworm and control using targeted insecticides or neem extracts during early whorl stages."
    },
    {
        "keywords": "poultry chicken broiler layer bird feed coccidiosis newcastle disease vaccination management biosecurity litter",
        "content": "Poultry management requires strict vaccination against Newcastle disease (using La Sota) and biosecurity to prevent Coccidiosis outbreaks. Maintain dry litter conditions, provide balanced stage-specific feed (starter, grower, finisher, layer mash), and enforce strict stocking densities."
    },
    {
        "keywords": "weed herbicide chemical control spear grass pre-emergence post-emergence manual clearing land preparation",
        "content": "Weed management combines manual clearing and approved selective/non-selective herbicides. Use glyphosate for pre-planting land preparation and targeted pre-emergence herbicides immediately after planting before crop germination to suppress aggressive grasses like spear grass."
    },
    {
        "keywords": "fertilizer npk urea application timing method split application ring method leaching micro-nutrients",
        "content": "Fertilizer application should utilize the ring or band method near the crop base followed by light soil incorporation. Split application (basal and top-dressing) maximizes nutrient absorption and prevents leaching caused by heavy tropical downpours."
    },
    {
        "keywords": "cassava mosaic disease cmd bacterial blight root rot iita resistant varieties tme 419 nr 8082 stem cuttings whitefly",
        "content": "Cassava production relies on planting certified disease-resistant IITA varieties (TME 419, NR 8082) to combat Cassava Mosaic Disease (CMD) and Cassava Bacterial Blight (CBB). Source clean stem cuttings and rogue out infected plants immediately to prevent whitefly vector transmission."
    },
    {
        "keywords": "yam tuber rot fusarium minisetting wood ash mancozeb seed treatment drainage ridges guinea savanna staking",
        "content": "Yam cultivation requires well-drained loamy soils. Treat cut seed yams using a slurry of clean wood ash and Mancozeb fungicide to prevent Fusarium and Aspergillus rot. Plant on high, well-drained ridges and provide sturdy staking for vines."
    },
    {
        "keywords": "rice yellow mottle virus rymv fadama lowland swamp water management vector control resistant varieties",
        "content": "Rice Yellow Mottle Virus (RYMV) affects lowland and Fadama systems. Manage using resistant varieties (FARO 44, NERICA), controlled paddy water drainage to disrupt vector beetles, and thorough cleaning of harvesting equipment."
    }
]

def build_and_merge(json_path="master_agro_kb.json", csv_path="naerls_verified.csv"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(MASTER_DATA, f, indent=4)
    print(f"External master file saved successfully: {json_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Target CSV {csv_path} not found in directory.")

    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["keywords", "content"])
        for record in MASTER_DATA:
            writer.writerow(record)
    print(f"Successfully merged {len(MASTER_DATA)} master records into {csv_path}!")

if __name__ == "__main__":
    build_and_merge()
