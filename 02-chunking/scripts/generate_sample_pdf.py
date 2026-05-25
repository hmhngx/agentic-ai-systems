"""Generate sample_doc.pdf — multi-chapter benchmark document for 02-chunking."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path(__file__).resolve().parent.parent / "sample_doc.pdf"

CHAPTERS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    (
        "Chapter 1: Machine Learning Foundations",
        [
            (
                "Supervised and Unsupervised Learning",
                [
                    (
                        "Machine learning systems learn statistical patterns from historical data "
                        "without being explicitly programmed for every rule. Supervised learning "
                        "maps labeled inputs to targets using regression and classification "
                        "algorithms trained on curated datasets. Engineers evaluate models with "
                        "held-out validation sets to estimate generalization before deployment. "
                        "Feature stores and pipelines ensure training and inference see consistent "
                        "transformations so production behavior matches offline experiments."
                    ),
                    (
                        "Unsupervised learning discovers structure in data that lacks labels. "
                        "Clustering groups similar records while dimensionality reduction compresses "
                        "high-dimensional vectors into interpretable projections. Anomaly detection "
                        "flags outliers in telemetry streams for security and reliability teams. "
                        "These methods power recommendation cold-start strategies and exploratory "
                        "analysis when annotation budgets are limited across large enterprise catalogs."
                    ),
                    (
                        "Semi-supervised and self-supervised techniques blend small labeled sets "
                        "with abundant unlabeled corpora. Contrastive learning pulls augmented views "
                        "of the same example closer in embedding space while pushing negatives apart. "
                        "Pretrained representations transfer to downstream tasks with modest fine-tuning "
                        "cost, which is why foundation models dominate modern natural language and vision "
                        "pipelines in both research prototypes and regulated production environments."
                    ),
                ],
            ),
            (
                "Deep Learning and Optimization",
                [
                    (
                        "Deep neural networks stack nonlinear layers to approximate complex functions. "
                        "Convolutional architectures exploit spatial locality in images while recurrent "
                        "and attention-based models capture sequential dependencies in text and time series. "
                        "Backpropagation computes gradients efficiently through computational graphs so "
                        "optimizers can adjust millions of parameters per iteration on accelerator hardware."
                    ),
                    (
                        "Stochastic gradient descent and its adaptive variants balance convergence speed "
                        "with stability across noisy mini-batches. Learning rate schedules warm up training "
                        "before decaying to fine-tune weights near minima. Batch normalization and layer "
                        "normalization reduce internal covariate shift, allowing deeper stacks to train "
                        "without vanishing or exploding activations that plagued early multilayer perceptrons."
                    ),
                    (
                        "Regularization through weight decay, dropout, and early stopping mitigates "
                        "overfitting when capacity exceeds data diversity. Data augmentation synthesizes "
                        "plausible variations that teach invariances without collecting new labels. "
                        "Monitoring calibration, fairness, and drift metrics in production closes the loop "
                        "between offline benchmarks and the lived experience of end users relying on predictions."
                    ),
                ],
            ),
            (
                "MLOps and Responsible Deployment",
                [
                    (
                        "Machine learning operations orchestrate training, versioning, and rollout of models "
                        "as first-class software artifacts. Containerized training jobs write metrics to experiment "
                        "trackers while artifact registries store weights, configs, and evaluation reports. "
                        "Canary releases compare shadow traffic before promoting a candidate model to serve "
                        "the majority of requests in latency-sensitive recommendation and ranking services."
                    ),
                    (
                        "Feature drift and concept drift detectors alert teams when input distributions or "
                        "label semantics shift after deployment. Retraining policies define triggers based on "
                        "calendar schedules, performance thresholds, or manual approval workflows in regulated "
                        "industries. Documentation of intended use, limitations, and failure modes supports "
                        "auditors reviewing high-stakes credit, healthcare, and safety-critical decision systems."
                    ),
                    (
                        "Human-in-the-loop review queues sample uncertain predictions for expert correction, "
                        "feeding active learning loops that prioritize informative examples. Privacy-preserving "
                        "techniques such as differential privacy and federated learning reduce exposure of "
                        "sensitive records while still improving global models across institutions that cannot "
                        "centralize raw data due to policy, sovereignty, or competitive constraints in practice."
                    ),
                ],
            ),
        ],
    ),
    (
        "Chapter 2: Ocean Biology and Marine Ecosystems",
        [
            (
                "Plankton and Primary Production",
                [
                    (
                        "Phytoplankton form the base of marine food webs and produce a large fraction of "
                        "Earth's oxygen through photosynthesis in sunlit surface waters. Diatoms and "
                        "dinoflagellates bloom when nutrients upwell along coasts, coloring satellite "
                        "imagery and supporting fisheries that depend on healthy zooplankton grazers. "
                        "Scientists measure chlorophyll concentrations to track seasonal cycles and "
                        "climate-driven shifts in productivity across ocean basins from pole to pole."
                    ),
                    (
                        "Zooplankton transport energy upward as they are consumed by fish larvae, jellyfish, "
                        "and filter-feeding whales. Vertical migration moves organisms hundreds of meters "
                        "each day, redistributing carbon and nutrients through the water column. Laboratory "
                        "experiments combined with autonomous floats reveal how acidification and warming "
                        "alter reproduction rates and lipid storage in species adapted to narrow temperature ranges."
                    ),
                    (
                        "Microbial loops recycle dissolved organic matter that would otherwise sink unused, "
                        "linking bacteria and viruses to higher trophic levels in ways classical food-chain "
                        "diagrams oversimplify. Metagenomic sequencing catalogs vast diversity in a single "
                        "liter of seawater, informing models of biogeochemical cycles that climate simulations "
                        "embed when projecting future ocean carbon uptake and oxygen minimum zone expansion."
                    ),
                ],
            ),
            (
                "Coral Reefs and Coastal Habitats",
                [
                    (
                        "Coral polyps host symbiotic algae that supply energy through photosynthesis while "
                        "building calcium carbonate skeletons that form reef structures sheltering thousands "
                        "of species. Bleaching occurs when thermal stress evicts algae, leaving corals pale "
                        "and vulnerable unless conditions improve quickly enough for recovery. Marine protected "
                        "areas and local fishing regulations aim to reduce additional stress from pollution "
                        "and overharvesting that compound climate impacts on shallow tropical ecosystems."
                    ),
                    (
                        "Mangroves and seagrass meadows stabilize shorelines, sequester carbon, and nursery "
                        "juvenile fish that later migrate to open ocean habitats. Restoration projects transplant "
                        "seedlings and monitor survival using acoustic telemetry and drone surveys. Coastal "
                        "engineers increasingly design hybrid infrastructure that allows wetland migration "
                        "inland as sea levels rise rather than relying solely on hardened seawalls that "
                        "disrupt sediment transport and larval settlement patterns along developed coastlines."
                    ),
                    (
                        "Invasive species and algal overgrowth can transform reef communities within seasons, "
                        "outcompeting native corals and herbivorous fish that maintain balance. Citizen science "
                        "programs train divers to record benthic cover categories standardized across regions, "
                        "building longitudinal datasets that policymakers use when negotiating international "
                        "conservation agreements and funding reef restoration at scales visible from space."
                    ),
                ],
            ),
            (
                "Deep Sea and Polar Biology",
                [
                    (
                        "Hydrothermal vents on mid-ocean ridges support chemosynthetic bacteria that sustain "
                        "tube worms, crabs, and snails without sunlight, revealing how life exploits chemical "
                        "gradients in extreme pressure and temperature. Submersibles map vent fields while "
                        "samplers preserve RNA for studying adaptation to toxic hydrogen sulfide plumes that "
                        "would kill surface organisms accustomed to oxygen-rich surface waters and photosynthesis."
                    ),
                    (
                        "Polar ecosystems revolve around sea ice seasonality that schedules plankton blooms "
                        "and penguin, seal, and krill population dynamics. Retreating ice changes albedo, "
                        "predator access, and breeding habitat, cascading through food webs monitored by "
                        "international research stations and indigenous knowledge holders who observe changes "
                        "across generations of subsistence hunting and coastal travel along Arctic shorelines."
                    ),
                    (
                        "Abyssal plains receive marine snow—aggregates of dead plankton and fecal pellets—that "
                        "fuel sparse but diverse communities of deposit feeders and scavengers. Long-term "
                        "sediment traps quantify carbon burial rates relevant to climate models. Mining "
                        "proposals for polymetallic nodules raise governance questions about disturbance "
                        "recovery timescales that may exceed human planning horizons in the deep Pacific basin."
                    ),
                ],
            ),
        ],
    ),
    (
        "Chapter 3: Ancient History and Civilizations",
        [
            (
                "Bronze Age Trade and Urbanization",
                [
                    (
                        "Early urban centers in Mesopotamia coordinated irrigation, temple administration, "
                        "and cuneiform record keeping that tracked grain loans and labor obligations. "
                        "Merchants moved lapis lazuli, timber, and copper along river routes and overland "
                        "caravans linking Egypt, Anatolia, and the Indus region in networks archaeologists "
                        "reconstruct from shipwreck cargoes and seal impressions on clay tablets stored in museums."
                    ),
                    (
                        "Palace economies redistributed staples to artisans and soldiers while legitimizing "
                        "rulers through monumental architecture and ritual calendars tied to agricultural seasons. "
                        "Law codes inscribed on stone proclaimed uniform penalties and property rights, "
                        "stabilizing commerce across diverse populations speaking unrelated languages yet "
                        "sharing markets, weights, and measures standardized by central authorities in capitals."
                    ),
                    (
                        "Collapse narratives for late Bronze Age states emphasize climate stress, invasion, "
                        "and systems fragility rather than single causes. Destruction layers and abandonment "
                        "horizons appear synchronously in some regions but not others, prompting debate about "
                        "teleconnections versus local failures. Modern resilience planners study these histories "
                        "when designing redundant supply chains that avoid overdependence on few trade partners."
                    ),
                ],
            ),
            (
                "Classical Mediterranean Empires",
                [
                    (
                        "Greek city-states experimented with citizenship, drama, and philosophy while competing "
                        "through colonization and naval warfare that spread the alphabet and coinage. Roman "
                        "expansion absorbed Hellenistic kingdoms, building roads, aqueducts, and legal institutions "
                        "that outlasted individual emperors. Provincial elites adopted Latin for administration "
                        "yet retained local cults and languages in villages supplying olive oil, grain, and metals."
                    ),
                    (
                        "Republican institutions gave way to imperial bureaucracy managing taxation, census, "
                        "and frontier defense along Rhine, Danube, and Euphrates borders. Archaeology of "
                        "Pompeii freezes daily life in volcanic ash, revealing bakeries, brothels, and electoral "
                        "graffiti that humanize abstract narratives drawn from elite literary sources biased toward "
                        "senatorial perspectives and moralizing historians writing generations after contested events."
                    ),
                    (
                        "Christianization and administrative division transformed the late empire as moving "
                        "capitals eastward concentrated resources near richer eastern provinces. Germanic successor "
                        "kingdoms blended Roman law with tribal custom, preserving fragments of archives that "
                        "Carolingian scholars later copied in monasteries, unintentionally preserving texts that "
                        "would shape medieval education and Renaissance rediscovery of classical rhetoric and science."
                    ),
                ],
            ),
            (
                "Asia and the Americas Before 1500",
                [
                    (
                        "Han and Tang China developed examination systems, paper currency experiments, and "
                        "Silk Road exchanges linking Chang'an to Samarkand and Baghdad. Buddhist monasteries "
                        "and Confucian academies coexisted with imperial edicts regulating land tenure and "
                        "corvee labor. Shipbuilding advances enabled voyages that distributed ceramics and "
                        "ideas across Southeast Asian archipelagos long before European oceanic circumnavigation."
                    ),
                    (
                        "In the Americas, Maya city-states mastered calendrics and hydraulic agriculture in "
                        "lowland rainforests while Andean polities terraced mountainsides and managed llama "
                        "caravans across ecological zones. Inka quipu knotted cords encoded administrative "
                        "information debated by scholars comparing them to writing systems elsewhere. "
                        "Mississippian mound builders networked river valleys with shared iconography and "
                        "ritual goods traded over hundreds of kilometers inland from Gulf and Atlantic coasts."
                    ),
                    (
                        "Historians combine radiocarbon dates, pollen cores, and ice isotopes to reconstruct "
                        "climate contexts for migration, warfare, and agricultural innovation. Decolonized "
                        "frameworks center indigenous agency rather than treating pre-contact societies as "
                        "static backdrops to modernity. Public archaeology engages descendant communities "
                        "when interpreting burial sites, repatriating artifacts, and designing museum exhibits "
                        "that acknowledge ongoing cultural continuity rather than extinct civilizations alone."
                    ),
                ],
            ),
        ],
    ),
    (
        "Chapter 4: Cooking Techniques and Food Science",
        [
            (
                "Heat Transfer and Flavor Development",
                [
                    (
                        "Cooking transforms ingredients through conduction, convection, and radiation, each "
                        "mechanism dominating different equipment from cast iron skillets to convection ovens. "
                        "Maillard reactions between amino acids and reducing sugars create browning aromas on "
                        "seared meats and toasted bread crusts distinct from caramelization of sugars alone. "
                        "Chefs layer seasoning throughout preparation so flavors integrate rather than sitting "
                        "only on surfaces where diners experience sharp contrasts instead of harmonious depth."
                    ),
                    (
                        "Sautéing preserves texture and color in vegetables by using high heat and minimal "
                        "moisture, while braising breaks down collagen in tough cuts through long moist cooking "
                        "at gentle temperatures. Blanching sets bright pigments in greens before shocking in ice "
                        "water halts enzymatic browning. Understanding carryover cooking prevents overcooking "
                        "proteins that continue rising in temperature after removal from heat sources in busy kitchens."
                    ),
                    (
                        "Acid, salt, fat, and umami balance structures tasting menus and home recipes alike. "
                        "Emulsions such as mayonnaise and hollandaise require stable interfaces between immiscible "
                        "phases controlled by lecithin and vigorous whisking technique. Fermentation generates "
                        "complexity in kimchi, miso, and sourdough through microbial metabolism monitored for "
                        "safety with pH and salinity thresholds that suppress pathogens while favoring desirable cultures."
                    ),
                ],
            ),
            (
                "Baking and Pastry Fundamentals",
                [
                    (
                        "Baking demands precise ratios of flour, water, fat, and leavening because gluten "
                        "networks trap gases produced by yeast or chemical reactions. Autolyse resting hydrates "
                        "flour before kneading, improving extensibility in artisan loaves baked on steel stones "
                        "that mimic hearth thermal mass. Laminated doughs fold butter layers that steam apart "
                        "into hundreds of flaky sheets in croissants and puff pastry enjoyed worldwide."
                    ),
                    (
                        "Sugar concentration affects boiling points in candy making, distinguishing thread, soft "
                        "ball, and hard crack stages measured with thermometers when humidity threatens consistency. "
                        "Chocolate tempering aligns cocoa butter crystals for snap and gloss on confections displayed "
                        "in shop windows. Egg proteins coagulate across narrow temperature bands, explaining why "
                        "custards require gentle water baths and constant stirring to avoid curdling into grainy textures."
                    ),
                    (
                        "Scaling recipes from home to commercial production introduces mixing time, oven airflow, "
                        "and hydration adjustments verified through pilot batches and sensory panels. Food safety "
                        "programs document hazard analysis for allergens, cross-contamination, and cooling curves "
                        "mandated by health departments. Mise en place organizes measured ingredients so service "
                        "lines execute consistently during peak dining hours without sacrificing quality for speed."
                    ),
                ],
            ),
            (
                "Global Cuisines and Technique Adaptation",
                [
                    (
                        "Stir-frying in woks depends on btu-rich burners and mise en place because cooking "
                        "completes in minutes. Indian tempering blooms spices in hot oil to release fat-soluble "
                        "aromas before incorporating aromatics into dals and curries. Japanese knife skills "
                        "respect grain direction in fish butchery, yielding clean cuts that preserve texture in "
                        "sashimi and nigiri presentations valued for precision and respect toward ingredients."
                    ),
                    (
                        "Mexican nixtamalization treats maize with alkali to improve nutrition and flavor in "
                        "tortillas and tamales central to regional identity. Mediterranean grilling over wood "
                        "imparts smoke compounds balanced by herb marinades and acidic finishing oils. "
                        "Technique migration through migration and media spreads methods globally, yet terroir "
                        "and local ingredients keep traditions recognizable to communities defending culinary heritage."
                    ),
                    (
                        "Modernist cuisine applies vacuum sealing, controlled temperature baths, and hydrocolloids "
                        "to achieve textures impossible with classical methods alone. Critics debate whether "
                        "technology distances diners from craft, while proponents argue reproducibility elevates "
                        "standards in fine dining laboratories publishing charts correlating time, temperature, "
                        "and texture outcomes for proteins, starches, and gels studied with rheometers in test kitchens."
                    ),
                ],
            ),
        ],
    ),
    (
        "Chapter 5: Urban Architecture and the Built Environment",
        [
            (
                "Modernism and the Industrial City",
                [
                    (
                        "Twentieth-century modernism promoted functional zoning, steel frames, and curtain walls "
                        "that freed facades from load-bearing masonry. Architects like Le Corbusier theorized "
                        "radiant cities with highways and towers in parks, influencing public housing projects "
                        "worldwide. Critics later documented how superblocks isolated residents from street life "
                        "and small businesses, prompting revisions that reintroduce mixed uses and human-scale "
                        "proportion in urban design guidelines adopted by progressive municipalities today."
                    ),
                    (
                        "Brutalism exposed raw concrete as an honest material expression of structure, producing "
                        "civic buildings and universities with dramatic sculptural forms now subject to preservation "
                        "debates. Energy crises in the 1970s pushed insulation standards and solar orientation studies "
                        "into building codes. Postmodern reactions reintroduced ornament, historical reference, and "
                        "color while questioning universal modernist claims about progress and rational planning alone."
                    ),
                    (
                        "Transit-oriented development clusters density near rail corridors to reduce automobile "
                        "dependence and land consumption at peripheries. Zoning reforms allow accessory dwelling units "
                        "and duplex conversions in single-family neighborhoods facing housing shortages. "
                        "Lifecycle assessments compare embodied carbon in steel, concrete, and timber systems, "
                        "informing policies that incentivize mass timber high-rises where fire codes and supply chains mature."
                    ),
                ],
            ),
            (
                "Sustainable and Resilient Design",
                [
                    (
                        "Green roofs manage stormwater, reduce heat island effects, and provide habitat in dense "
                        "districts where ground-level parks are scarce. Passive house standards minimize heating and "
                        "cooling loads through airtight envelopes, heat recovery ventilation, and high-performance "
                        "glazing oriented for solar gain in cold climates and shading in warm ones. Municipalities "
                        "benchmark energy use in large buildings, publishing disclosures that influence leasing decisions."
                    ),
                    (
                        "Flood resilience elevates mechanical systems, installs deployable barriers, and preserves "
                        "waterfront parks that flood by design rather than channelizing rivers solely with concrete. "
                        "Earthquake-resistant detailing ductile frames and base isolation in critical facilities like "
                        "hospitals. Wildfire-prone regions specify ember-resistant vents and defensible space regulations "
                        "coordinated with forestry management beyond individual parcel aesthetics alone."
                    ),
                    (
                        "Adaptive reuse converts warehouses, churches, and offices into housing and cultural venues, "
                        "avoiding demolition waste and honoring neighborhood memory. Digital twins simulate occupancy, "
                        "HVAC performance, and maintenance backlogs using sensor feeds from smart buildings. "
                        "Participatory design workshops incorporate residents' knowledge of microclimates, noise, and "
                        "social networks that pure top-down master plans often miss when reshaping districts under renewal."
                    ),
                ],
            ),
            (
                "Public Space and Urban Experience",
                [
                    (
                        "Streets are contested spaces balancing vehicles, cyclists, pedestrians, and outdoor dining "
                        "expanded after pandemic-era experiments. Complete streets policies add protected bike lanes "
                        "and curb ramps mandated by accessibility law. Wayfinding signage and lighting influence "
                        "perceived safety, especially for women and elders traveling at night through districts "
                        "revitalized with arts programming and local retail rather than blank frontages."
                    ),
                    (
                        "Plazas and parks host protest, celebration, and daily leisure that define democratic urban "
                        "life beyond commercial transactions. Landscape architects shape topography, planting palettes, "
                        "and water features that mitigate noise and heat while supporting biodiversity corridors linking "
                        "fragmented habitats. Maintenance funding often determines whether visionary renderings survive "
                        "budget cuts that leave fountains dry and planting beds weedy within a few seasons of opening."
                    ),
                    (
                        "Historic preservation districts protect character through design review while debating how "
                        "to accommodate solar panels, mobility devices, and net-zero retrofits without destroying "
                        "heritage fabric. Global tourism pressures convert centers into short-term rental markets, "
                        "displacing longtime residents unless rent stabilization and community land trusts intervene. "
                        "Architects increasingly co-design with planners, ecologists, and social workers to deliver "
                        "infrastructure that is legible, equitable, and resilient across climate uncertainties ahead."
                    ),
                ],
            ),
        ],
    ),
]


def build_story() -> list:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=14,
        spaceBefore=6,
    )
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=10,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=11,
        leading=14,
        spaceAfter=10,
    )

    story: list = []
    for ch_idx, (chapter_title, sections) in enumerate(CHAPTERS):
        if ch_idx > 0:
            story.append(PageBreak())
        story.append(Paragraph(chapter_title, title_style))
        story.append(Spacer(1, 0.15 * inch))
        for section_title, paragraphs in sections:
            story.append(Paragraph(f"## {section_title}", section_style))
            for para in paragraphs:
                story.append(Paragraph(para, body_style))
    return story


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(build_story())
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
