export const EXPANSIONS = [
  'corpera',
  'promo',
  'venus',
  'colonies',
  'prelude',
  'prelude2',
  'turmoil',
  'community',
  'ares',
  'moon',
  'pathfinders',
  'ceo',
  'starwars',
  'underworld',
] as const;

export const GAME_MODULES = [
  'base',
  ...EXPANSIONS,
] as const;
export type GameModule = typeof GAME_MODULES[number];

export type Expansion = Exclude<GameModule, 'base'>;

export enum CardName {
  // Standard projects:
  SELL_PATENTS_STANDARD_PROJECT = 'Sell Patents',
  POWER_PLANT_STANDARD_PROJECT = 'Power Plant:SP',
  ASTEROID_STANDARD_PROJECT = 'Asteroid:SP',
  BUFFER_GAS_STANDARD_PROJECT = 'Buffer Gas',
  BUILD_COLONY_STANDARD_PROJECT = 'Colony',
  AQUIFER_STANDARD_PROJECT = 'Aquifer',
  GREENERY_STANDARD_PROJECT = 'Greenery',
  CITY_STANDARD_PROJECT = 'City',
  AIR_SCRAPPING_STANDARD_PROJECT = 'Air Scrapping',
  AIR_SCRAPPING_STANDARD_PROJECT_VARIANT = 'Air Scrapping (Var)',

  // Standard actions:
  CONVERT_PLANTS = 'Convert Plants',
  CONVERT_HEAT = 'Convert Heat',

  ACQUIRED_COMPANY = 'Acquired Company',
  ADAPTATION_TECHNOLOGY = 'Adaptation Technology',
  ADAPTED_LICHEN = 'Adapted Lichen',
  ADVANCED_ALLOYS = 'Advanced Alloys',
  ADVANCED_ECOSYSTEMS = 'Advanced Ecosystems',
  AEROBRAKED_AMMONIA_ASTEROID = 'Aerobraked Ammonia Asteroid',
  AI_CENTRAL = 'AI Central',
  AIR_RAID = 'Air Raid',
  AIRLINERS = 'Airliners',
  ALGAE = 'Algae',
  ANTI_GRAVITY_TECHNOLOGY = 'Anti-Gravity Technology',
  ANTS = 'Ants',
  AQUIFER_PUMPING = 'Aquifer Pumping',
  AQUIFER_TURBINES = 'Aquifer Turbines',
  ARCHAEBACTERIA = 'ArchaeBacteria',
  ARTIFICIAL_LAKE = 'Artificial Lake',
  ARTIFICIAL_PHOTOSYNTHESIS = 'Artificial Photosynthesis',
  ARCTIC_ALGAE = 'Arctic Algae',
  ASTEROID = 'Asteroid',
  ASTEROID_MINING = 'Asteroid Mining',
  ASTEROID_MINING_CONSORTIUM = 'Asteroid Mining Consortium',
  ATMO_COLLECTORS = 'Atmo Collectors',
  BREATHING_FILTERS = 'Breathing Filters',
  BRIBED_COMMITTEE = 'Bribed Committee',
  BEAM_FROM_A_THORIUM_ASTEROID = 'Beam From A Thorium Asteroid',
  BIG_ASTEROID = 'Big Asteroid',
  BIOMASS_COMBUSTORS = 'Biomass Combustors',
  BIRDS = 'Birds',
  BLACK_POLAR_DUST = 'Black Polar Dust',
  BUILDING_INDUSTRIES = 'Building Industries',
  BUSHES = 'Bushes',
  BUSINESS_CONTACTS = 'Business Contacts',
  BUSINESS_NETWORK = 'Business Network',
  CALLISTO_PENAL_MINES = 'Callisto Penal Mines',
  CARBONATE_PROCESSING = 'Carbonate Processing',
  CAPITAL = 'Capital',
  CARETAKER_CONTRACT = 'Caretaker Contract',
  CARTEL = 'Cartel',
  CEOS_FAVORITE_PROJECT = 'CEO\'s Favorite Project',
  CLOUD_SEEDING = 'Cloud Seeding',
  COLONIZER_TRAINING_CAMP = 'Colonizer Training Camp',
  COMET = 'Comet',
  COMMERCIAL_DISTRICT = 'Commercial District',
  COMMUNITY_SERVICES = 'Community Services',
  CONSCRIPTION = 'Conscription',
  CONVOY_FROM_EUROPA = 'Convoy From Europa',
  CORONA_EXTRACTOR = 'Corona Extractor',
  CORPORATE_STRONGHOLD = 'Corporate Stronghold',
  CRYO_SLEEP = 'Cryo-Sleep',
  CUPOLA_CITY = 'Cupola City',
  DECOMPOSERS = 'Decomposers',
  DEEP_WELL_HEATING = 'Deep Well Heating',
  DEIMOS_DOWN = 'Deimos Down',
  DESIGNED_MICROORGANISMS = 'Designed Microorganisms',
  DEVELOPMENT_CENTER = 'Development Center',
  DIRIGIBLES = 'Dirigibles',
  DOME_FARMING = 'Dome Farming',
  DOMED_CRATER = 'Domed Crater',
  DUST_SEALS = 'Dust Seals',
  EARLY_SETTLEMENT = 'Early Settlement',
  EARTH_CATAPULT = 'Earth Catapult',
  EARTH_ELEVATOR = 'Earth Elevator',
  EARTH_OFFICE = 'Earth Office',
  ECCENTRIC_SPONSOR = 'Eccentric Sponsor',
  ECOLOGICAL_ZONE = 'Ecological Zone',
  ECOLOGY_EXPERTS = 'Ecology Experts',
  ECOLOGY_RESEARCH = 'Ecology Research',
  ELECTRO_CATAPULT = 'Electro Catapult',
  ENERGY_SAVING = 'Energy Saving',
  ENERGY_TAPPING = 'Energy Tapping',
  EOS_CHASMA_NATIONAL_PARK = 'Eos Chasma National Park',
  EQUATORIAL_MAGNETIZER = 'Equatorial Magnetizer',
  EXTREME_COLD_FUNGUS = 'Extreme-Cold Fungus',
  FARMING = 'Farming',
  FISH = 'Fish',
  FLOATER_LEASING = 'Floater Leasing',
  FLOATER_PROTOTYPES = 'Floater Prototypes',
  FLOATER_TECHNOLOGY = 'Floater Technology',
  FLOODING = 'Flooding',
  FOOD_FACTORY = 'Food Factory',
  FUEL_FACTORY = 'Fuel Factory',
  FUELED_GENERATORS = 'Fueled Generators',
  FUSION_POWER = 'Fusion Power',
  GALILEAN_WAYSTATION = 'Galilean Waystation',
  GANYMEDE_COLONY = 'Ganymede Colony',
  GENE_REPAIR = 'Gene Repair',
  GEOTHERMAL_POWER = 'Geothermal Power',
  GHG_PRODUCING_BACTERIA = 'GHG Producing Bacteria',
  GHG_FACTORIES = 'GHG Factories',
  GIANT_ICE_ASTEROID = 'Giant Ice Asteroid',
  GIANT_SPACE_MIRROR = 'Giant Space Mirror',
  GRASS = 'Grass',
  GREAT_AQUIFER = 'Great Aquifer',
  GREAT_DAM = 'Great Dam',
  GREAT_ESCARPMENT_CONSORTIUM = 'Great Escarpment Consortium',
  GREENHOUSES = 'Greenhouses',
  GYROPOLIS = 'Gyropolis',
  HACKERS = 'Hackers',
  HEATHER = 'Heather',
  HEAT_TRAPPERS = 'Heat Trappers',
  HEAVY_TAXATION = 'Heavy Taxation',
  HERBIVORES = 'Herbivores',
  HIRED_RAIDERS = 'Hired Raiders',
  HOUSE_PRINTING = 'House Printing',
  ICE_ASTEROID = 'Ice Asteroid',
  ICE_CAP_MELTING = 'Ice Cap Melting',
  ICE_MOON_COLONY = 'Ice Moon Colony',
  IMMIGRANT_CITY = 'Immigrant City',
  IMMIGRATION_SHUTTLES = 'Immigration Shuttles',
  IMPACTOR_SWARM = 'Impactor Swarm',
  IMPORTED_GHG = 'Imported GHG',
  IMPORTED_HYDROGEN = 'Imported Hydrogen',
  IMPORTED_NITROGEN = 'Imported Nitrogen',
  IMPORT_OF_ADVANCED_GHG = 'Import of Advanced GHG',
  INDENTURED_WORKERS = 'Indentured Workers',
  INDUSTRIAL_MICROBES = 'Industrial Microbes',
  INSECTS = 'Insects',
  INSULATION = 'Insulation',
  INTERPLANETARY_COLONY_SHIP = 'Interplanetary Colony Ship',
  INTERSTELLAR_COLONY_SHIP = 'Interstellar Colony Ship',
  INVENTION_CONTEST = 'Invention Contest',
  INVENTORS_GUILD = 'Inventors\' Guild',
  INVESTMENT_LOAN = 'Investment Loan',
  IO_MINING_INDUSTRIES = 'Io Mining Industries',
  IRONWORKS = 'Ironworks',
  JOVIAN_LANTERNS = 'Jovian Lanterns',
  JUPITER_FLOATING_STATION = 'Jupiter Floating Station',
  KELP_FARMING = 'Kelp Farming',
  LAGRANGE_OBSERVATORY = 'Lagrange Observatory',
  LAKE_MARINERIS = 'Lake Marineris',
  LAND_CLAIM = 'Land Claim',
  LARGE_CONVOY = 'Large Convoy',
  LAVA_FLOWS = 'Lava Flows',
  LAVA_TUBE_SETTLEMENT = 'Lava Tube Settlement',
  LICHEN = 'Lichen',
  LIGHTNING_HARVEST = 'Lightning Harvest',
  LIVESTOCK = 'Livestock',
  LOCAL_HEAT_TRAPPING = 'Local Heat Trapping',
  LUNAR_BEAM = 'Lunar Beam',
  LUNA_GOVERNOR = 'Luna Governor',
  LUNAR_EXPORTS = 'Lunar Exports',
  LUNAR_MINING = 'Lunar Mining',
  MAGNETIC_FIELD_DOME = 'Magnetic Field Dome',
  MAGNETIC_FIELD_GENERATORS = 'Magnetic Field Generators',
  MARKET_MANIPULATION = 'Market Manipulation',
  MARTIAN_INDUSTRIES = 'Martian Industries',
  MARTIAN_ZOO = 'Martian Zoo',
  MANGROVE = 'Mangrove',
  MARS_UNIVERSITY = 'Mars University',
  MARTIAN_RAILS = 'Martian Rails',
  MASS_CONVERTER = 'Mass Converter',
  MEDIA_ARCHIVES = 'Media Archives',
  MEDIA_GROUP = 'Media Group',
  MEDICAL_LAB = 'Medical Lab',
  METHANE_FROM_TITAN = 'Methane From Titan',
  MICRO_MILLS = 'Micro-Mills',
  MINE = 'Mine',
  MINERAL_DEPOSIT = 'Mineral Deposit',
  MIRANDA_RESORT = 'Miranda Resort',
  MINING_AREA = 'Mining Area',
  MINING_COLONY = 'Mining Colony',
  MINING_EXPEDITION = 'Mining Expedition',
  MINING_OPERATIONS = 'Mining Operations',
  MINING_QUOTA = 'Mining Quota',
  MINING_RIGHTS = 'Mining Rights',
  MINORITY_REFUGE = 'Minority Refuge',
  MOHOLE = 'Mohole',
  MOHOLE_AREA = 'Mohole Area',
  MOHOLE_EXCAVATION = 'Mohole Excavation',
  MOLECULAR_PRINTING = 'Molecular Printing',
  MOSS = 'Moss',
  NATURAL_PRESERVE = 'Natural Preserve',
  NITRITE_REDUCING_BACTERIA = 'Nitrite Reducing Bacteria',
  NITROGEN_RICH_ASTEROID = 'Nitrogen-Rich Asteroid',
  NITROGEN_FROM_TITAN = 'Nitrogen from Titan',
  NITROPHILIC_MOSS = 'Nitrophilic Moss',
  NOCTIS_CITY = 'Noctis City',
  NOCTIS_FARMING = 'Noctis Farming',
  NUCLEAR_POWER = 'Nuclear Power',
  NUCLEAR_ZONE = 'Nuclear Zone',
  OLYMPUS_CONFERENCE = 'Olympus Conference',
  OMNICOURT = 'Omnicourt',
  OPEN_CITY = 'Open City',
  OPTIMAL_AEROBRAKING = 'Optimal Aerobraking',
  ORE_PROCESSOR = 'Ore Processor',
  PERMAFROST_EXTRACTION = 'Permafrost Extraction',
  PEROXIDE_POWER = 'Peroxide Power',
  PETS = 'Pets',
  PHOBOS_SPACE_HAVEN = 'Phobos Space Haven',
  PHYSICS_COMPLEX = 'Physics Complex',
  PIONEER_SETTLEMENT = 'Pioneer Settlement',
  PLANTATION = 'Plantation',
  POLAR_INDUSTRIES = 'Polar Industries',
  POWER_GRID = 'Power Grid',
  POWER_INFRASTRUCTURE = 'Power Infrastructure',
  POWER_PLANT = 'Power Plant',
  POWER_SUPPLY_CONSORTIUM = 'Power Supply Consortium',
  PREDATORS = 'Predators',
  PRODUCTIVE_OUTPOST = 'Productive Outpost',
  PROTECTED_HABITATS = 'Protected Habitats',
  PROTECTED_VALLEY = 'Protected Valley',
  PSYCHROPHILES = 'Psychrophiles',
  QUANTUM_COMMUNICATIONS = 'Quantum Communications',
  QUANTUM_EXTRACTOR = 'Quantum Extractor',
  RAD_CHEM_FACTORY = 'Rad-Chem Factory',
  RAD_SUITS = 'Rad-Suits',
  RED_SPOT_OBSERVATORY = 'Red Spot Observatory',
  REFUGEE_CAMPS = 'Refugee Camps',
  REGOLITH_EATERS = 'Regolith Eaters',
  RELEASE_OF_INERT_GASES = 'Release of Inert Gases',
  RESEARCH = 'Research',
  RESEARCH_OUTPOST = 'Research Outpost',
  RESEARCH_COLONY = 'Research Colony',
  RESTRICTED_AREA = 'Restricted Area',
  ROBOTIC_WORKFORCE = 'Robotic Workforce',
  ROVER_CONSTRUCTION = 'Rover Construction',
  RIM_FREIGHTERS = 'Rim Freighters',
  SABOTAGE = 'Sabotage',
  SATELLITES = 'Satellites',
  SEARCH_FOR_LIFE = 'Search For Life',
  SECURITY_FLEET = 'Security Fleet',
  SELF_SUFFICIENT_SETTLEMENT = 'Self-Sufficient Settlement',
  SISTER_PLANET_SUPPORT = 'Sister Planet Support',
  SKY_DOCKS = 'Sky Docks',
  SMALL_ANIMALS = 'Small Animals',
  SOIL_FACTORY = 'Soil Factory',
  SOLAR_POWER = 'Solar Power',
  SOLAR_PROBE = 'Solar Probe',
  SOLAR_REFLECTORS = 'Solar Reflectors',
  SOLARNET = 'Solarnet',
  SPACE_ELEVATOR = 'Space Elevator',
  SPACE_PORT = 'Space Port',
  SPACE_PORT_COLONY = 'Space Port Colony',
  SPINOFF_DEPARTMENT = 'Spin-off Department',
  STRIP_MINE = 'Strip Mine',
  SUBTERRANEAN_RESERVOIR = 'Subterranean Reservoir',
  SUBZERO_SALT_FISH = 'Sub-zero Salt Fish',
  SHUTTLES = 'Shuttles',
  SOLAR_WIND_POWER = 'Solar Wind Power',
  SOLETTA = 'Soletta',
  SPACE_MIRRORS = 'Space Mirrors',
  SPACE_STATION = 'Space Station',
  SPECIAL_DESIGN = 'Special Design',
  SPONSORS = 'Sponsors',
  STEELWORKS = 'Steelworks',
  STANDARD_TECHNOLOGY = 'Standard Technology',
  SYMBIOTIC_FUNGUS = 'Symbiotic Fungus',
  TARDIGRADES = 'Tardigrades',
  TECHNOLOGY_DEMONSTRATION = 'Technology Demonstration',
  TECTONIC_STRESS_POWER = 'Tectonic Stress Power',
  TERRAFORMING_GANYMEDE = 'Terraforming Ganymede',
  TITAN_AIRSCRAPPING = 'Titan Air-scrapping',
  TITAN_FLOATING_LAUNCHPAD = 'Titan Floating Launch-pad',
  TITAN_SHUTTLES = 'Titan Shuttles',
  TITANIUM_MINE = 'Titanium Mine',
  TOLL_STATION = 'Toll Station',
  TOWING_A_COMET = 'Towing A Comet',
  TRADE_ENVOYS = 'Trade Envoys',
  TRADING_COLONY = 'Trading Colony',
  TRANS_NEPTUNE_PROBE = 'Trans-Neptune Probe',
  TREES = 'Trees',
  TROPICAL_RESORT = 'Tropical Resort',
  TUNDRA_FARMING = 'Tundra Farming',
  UNDERGROUND_CITY = 'Underground City',
  UNDERGROUND_DETONATIONS = 'Underground Detonations',
  URBAN_DECOMPOSERS = 'Urban Decomposers',
  URBANIZED_AREA = 'Urbanized Area',
  VESTA_SHIPYARD = 'Vesta Shipyard',
  VIRAL_ENHANCERS = 'Viral Enhancers',
  VIRUS = 'Virus',
  WARP_DRIVE = 'Warp Drive',
  WATER_IMPORT_FROM_EUROPA = 'Water Import From Europa',
  // Venus:
  AERIAL_MAPPERS = 'Aerial Mappers',
  AEROSPORT_TOURNAMENT = 'Aerosport Tournament',
  AIR_SCRAPPING_EXPEDITION = 'Air-Scrapping Expedition',
  APHRODITE = 'Aphrodite',
  ATALANTA_PLANITIA_LAB = 'Atalanta Planitia Lab',
  ATMOSCOOP = 'Atmoscoop',
  CELESTIC = 'Celestic',
  COMET_FOR_VENUS = 'Comet for Venus',
  CORRODER_SUITS = 'Corroder Suits',
  DAWN_CITY = 'Dawn City',
  DEUTERIUM_EXPORT = 'Deuterium Export',
  EXTRACTOR_BALLOONS = 'Extractor Balloons',
  EXTREMOPHILES = 'Extremophiles',
  FLOATING_HABS = 'Floating Habs',
  FORCED_PRECIPITATION = 'Forced Precipitation',
  FREYJA_BIODOMES = 'Freyja Biodomes',
  GHG_IMPORT_FROM_VENUS = 'GHG Import From Venus',
  GIANT_SOLAR_SHADE = 'Giant Solar Shade',
  HYDROGEN_TO_VENUS = 'Hydrogen to Venus',
  IO_SULPHUR_RESEARCH = 'Io Sulphur Research',
  ISHTAR_MINING = 'Ishtar Mining',
  JET_STREAM_MICROSCRAPPERS = 'Jet Stream Microscrappers',
  LOCAL_SHADING = 'Local Shading',
  LUNA_METROPOLIS = 'Luna Metropolis',
  LUXURY_FOODS = 'Luxury Foods',
  MAXWELL_BASE = 'Maxwell Base',
  MORNING_STAR_INC = 'Morning Star Inc.',
  NEUTRALIZER_FACTORY = 'Neutralizer Factory',
  ORBITAL_REFLECTORS = 'Orbital Reflectors',
  ROTATOR_IMPACTS = 'Rotator Impacts',
  SPIN_INDUCING_ASTEROID = 'Spin-Inducing Asteroid',
  SPONSORED_ACADEMIES = 'Sponsored Academies',
  STRATOPOLIS = 'Stratopolis',
  STRATOSPHERIC_BIRDS = 'Stratospheric Birds',
  SULPHUR_EATING_BACTERIA = 'Sulphur-Eating Bacteria',
  SULPHUR_EXPORTS = 'Sulphur Exports',
  TERRAFORMING_CONTRACT = 'Terraforming Contract',
  THERMOPHILES = 'Thermophiles',
  VENUS_GOVERNOR = 'Venus Governor',
  VENUSIAN_ANIMALS = 'Venusian Animals',
  VENUSIAN_INSECTS = 'Venusian Insects',
  VENUSIAN_PLANTS = 'Venusian Plants',
  VENUS_MAGNETIZER = 'Venus Magnetizer',
  VENUS_SOILS = 'Venus Soils',
  VENUS_WAYSTATION = 'Venus Waystation',
  VIRON = 'Viron',
  WATER_TO_VENUS = 'Water to Venus',
  WATER_SPLITTING_PLANT = 'Water Splitting Plant',
  WAVE_POWER = 'Wave Power',
  WINDMILLS = 'Windmills',
  WORMS = 'Worms',
  ZEPPELINS = 'Zeppelins',

  // Corps:
  BEGINNER_CORPORATION = 'Beginner Corporation',
  CREDICOR = 'CrediCor',
  ECOLINE = 'EcoLine',
  HELION = 'Helion',
  INTERPLANETARY_CINEMATICS = 'Interplanetary Cinematics',
  INVENTRIX = 'Inventrix',
  MINING_GUILD = 'Mining Guild',
  PHOBOLOG = 'PhoboLog',
  SATURN_SYSTEMS = 'Saturn Systems',
  TERACTOR = 'Teractor',
  THARSIS_REPUBLIC = 'Tharsis Republic',
  THORGATE = 'Thorgate',
  UNITED_NATIONS_MARS_INITIATIVE = 'United Nations Mars Initiative',
  ACQUIRED_SPACE_AGENCY = 'Acquired Space Agency',
  // Preludes:
  ALLIED_BANK = 'Allied Bank',
  BIOFUELS = 'Biofuels',
  BIOLAB = 'Biolab',
  BIOSPHERE_SUPPORT = 'Biosphere Support',
  BUSINESS_EMPIRE = 'Business Empire',
  CHEUNG_SHING_MARS = 'Cheung Shing MARS',
  DONATION = 'Donation',
  EXPERIMENTAL_FOREST = 'Experimental Forest',
  GALILEAN_MINING = 'Galilean Mining',
  HUGE_ASTEROID = 'Huge Asteroid',
  IO_RESEARCH_OUTPOST = 'Io Research Outpost',
  LOAN = 'Loan',
  MARTIAN_SURVEY = 'Martian Survey',
  METAL_RICH_ASTEROID = 'Metal-Rich Asteroid',
  METALS_COMPANY = 'Metals Company',
  NITROGEN_SHIPMENT = 'Nitrogen Shipment',
  ORBITAL_CONSTRUCTION_YARD = 'Orbital Construction Yard',
  POINT_LUNA = 'Point Luna',
  POWER_GENERATION = 'Power Generation',
  RESEARCH_COORDINATION = 'Research Coordination',
  RESEARCH_NETWORK = 'Research Network',
  ROBINSON_INDUSTRIES = 'Robinson Industries',
  SF_MEMORIAL = 'SF Memorial',
  SMELTING_PLANT = 'Smelting Plant',
  SOCIETY_SUPPORT = 'Society Support',
  SPACE_HOTELS = 'Space Hotels',
  SUPPLIER = 'Supplier',
  SUPPLY_DROP = 'Supply Drop',
  UNMI_CONTRACTOR = 'UNMI Contractor',
  VALLEY_TRUST = 'Valley Trust',
  VITOR = 'Vitor',
  ARIDOR = 'Aridor',
  ARKLIGHT = 'Arklight',
  POSEIDON = 'Poseidon',
  STORMCRAFT_INCORPORATED = 'Stormcraft Incorporated',
  ARCADIAN_COMMUNITIES = 'Arcadian Communities',
  ASTRODRILL = 'Astrodrill',
  ADVERTISING = 'Advertising',
  PHARMACY_UNION = 'Pharmacy Union',
  INDUSTRIAL_CENTER = 'Industrial Center',
  FACTORUM = 'Factorum',
  LAKEFRONT_RESORTS = 'Lakefront Resorts',
  TERRALABS_RESEARCH = 'Terralabs Research',
  SEPTUM_TRIBUS = 'Septem Tribus',
  MONS_INSURANCE = 'Mons Insurance',
  SPLICE = 'Splice',
  PHILARES = 'Philares',
  PRISTAR = 'Pristar',
  RECYCLON = 'Recyclon',
  UTOPIA_INVEST = 'Utopia Invest',
  MANUTECH = 'Manutech',
  SELF_REPLICATING_ROBOTS = 'Self-replicating Robots',
  POLYPHEMOS = 'Polyphemos',
  PENGUINS = 'Penguins',
  SMALL_ASTEROID = 'Small Asteroid',
  SNOW_ALGAE = 'Snow Algae',
  AERIAL_LENSES = 'Aerial Lenses',
  BANNED_DELEGATE = 'Banned Delegate',
  CULTURAL_METROPOLIS = 'Cultural Metropolis',
  DIASPORA_MOVEMENT = 'Diaspora Movement',
  EVENT_ANALYSTS = 'Event Analysts',
  GMO_CONTRACT = 'GMO Contract',
  MARTIAN_MEDIA_CENTER = 'Martian Media Center',
  PARLIAMENT_HALL = 'Parliament Hall',
  PR_OFFICE = 'PR Office',
  PUBLIC_CELEBRATIONS = 'Public Celebrations',
  RECRUITMENT = 'Recruitment',
  RED_TOURISM_WAVE = 'Red Tourism Wave',
  SPONSORED_MOHOLE = 'Sponsored Mohole',
  SUPPORTED_RESEARCH = 'Supported Research',
  WILDLIFE_DOME = 'Wildlife Dome',
  VOTE_OF_NO_CONFIDENCE = 'Vote Of No Confidence',

  // Prelude 2
  // Prelude 2 Project Cards
  CERES_TECH_MARKET = 'Ceres Tech Market',
  CLOUD_TOURISM = 'Cloud Tourism',
  COLONIAL_ENVOYS = 'Colonial Envoys',
  COLONIAL_REPRESENTATION = 'Colonial Representation',
  ENVOYS_FROM_VENUS = 'Envoys From Venus',
  FLOATING_REFINERY = 'Floating Refinery',
  FRONTIER_TOWN = 'Frontier Town',
  GHG_SHIPMENT = 'GHG Shipment',
  ISHTAR_EXPEDITION = 'Ishtar Expedition',
  JOVIAN_ENVOYS = 'Jovian Envoys',
  L1_TRADE_TERMINAL = 'L1 Trade Terminal',
  MICROGRAVITY_NUTRITION = 'Microgravity Nutrition',
  RED_APPEASEMENT = 'Red Appeasement',
  SOIL_STUDIES = 'Soil Studies',
  SPECIAL_PERMIT = 'Special Permit',
  SPONSORING_NATION = 'Sponsoring Nation',
  STRATOSPHERIC_EXPEDITION = 'Stratospheric Expedition',
  SUMMIT_LOGISTICS = 'Summit Logistics',
  UNEXPECTED_APPLICATION = 'Unexpected Application',
  VENUS_ALLIES = 'Venus Allies',
  VENUS_ORBITAL_SURVEY = 'Venus Orbital Survey',
  VENUS_SHUTTLES = 'Venus Shuttles',
  VENUS_TRADE_HUB = 'Venus Trade Hub',
  WG_PROJECT = 'WG Project',

  // Prelude 2 Preludes
  APPLIED_SCIENCE = 'Applied Science',
  ATMOSPHERIC_ENHANCERS = 'Atmospheric Enhancers',
  BOARD_OF_DIRECTORS = 'Board of Directors',
  COLONY_TRADE_HUB = 'Colony Trade Hub',
  CORRIDORS_OF_POWER = 'Corridors of Power',
  EARLY_COLONIZATION = 'Early Colonization',
  FLOATING_TRADE_HUB = 'Floating Trade Hub',
  FOCUSED_ORGANIZATION = 'Focused Organization',
  HIGH_CIRCLES = 'High Circles',
  INDUSTRIAL_COMPLEX = 'Industrial Complex',
  MAIN_BELT_ASTEROIDS = 'Main Belt Asteroids',
  NOBEL_PRIZE = 'Nobel Prize',
  OLD_MINING_COLONY = 'Old Mining Colony',
  PLANETARY_ALLIANCE = 'Planetary Alliance',
  PRESERVATION_PROGRAM = 'Preservation Program',
  PROJECT_EDEN = 'Project Eden',
  RECESSION = 'Recession',
  RISE_TO_POWER = 'Rise To Power',
  SOIL_BACTERIA = 'Soil Bacteria',
  SPACE_LANES = 'Space Lanes',
  SUITABLE_INFRASTRUCTURE = 'Suitable Infrastructure',
  TERRAFORMING_DEAL = 'Terraforming Deal',
  VENUS_CONTRACT = 'Venus Contract',
  VENUS_L1_SHADE = 'Venus L1 Shade',
  WORLD_GOVERNMENT_ADVISOR = 'World Government Advisor',

  // Prelude 2 Corps
  NIRGAL_ENTERPRISES = 'Nirgal Enterprises',
  PALLADIN_SHIPPING = 'Palladin Shipping',
  ECOTEC = 'EcoTec',
  SAGITTA_FRONTIER_SERVICES = 'Sagitta Frontier Services',
  SPIRE = 'Spire',

  // Promo cards
  DUSK_LASER_MINING = 'Dusk Laser Mining',
  PROJECT_INSPECTION = 'Project Inspection',
  ENERGY_MARKET = 'Energy Market',
  HI_TECH_LAB = 'Hi-Tech Lab',
  INTERPLANETARY_TRADE = 'Interplanetary Trade',
  LAW_SUIT = 'Law Suit',
  MERCURIAN_ALLOYS = 'Mercurian Alloys',
  ORBITAL_CLEANUP = 'Orbital Cleanup',
  POLITICAL_ALLIANCE = 'Political Alliance',
  REGO_PLASTICS = 'Rego Plastics',
  SATURN_SURFING = 'Saturn Surfing',
  STANFORD_TORUS = 'Stanford Torus',
  ASTEROID_HOLLOWING = 'Asteroid Hollowing',
  COMET_AIMING = 'Comet Aiming',
  CRASH_SITE_CLEANUP = 'Crash Site Cleanup',
  CUTTING_EDGE_TECHNOLOGY = 'Cutting Edge Technology',
  DIRECTED_IMPACTORS = 'Directed Impactors',
  FIELD_CAPPED_CITY = 'Field-Capped City',
  MAGNETIC_SHIELD = 'Magnetic Shield',
  MELTWORKS = 'Meltworks',
  MOHOLE_LAKE = 'Mohole Lake',
  DIVERSITY_SUPPORT = 'Diversity Support',
  JOVIAN_EMBASSY = 'Jovian Embassy',
  TOPSOIL_CONTRACT = 'Topsoil Contract',
  IMPORTED_NUTRIENTS = 'Imported Nutrients',
  ASTEROID_DEFLECTION_SYSTEM = 'Asteroid Deflection System',
  SUB_CRUST_MEASUREMENTS = 'Sub-Crust Measurements',
  POTATOES = 'Potatoes',
  MEAT_INDUSTRY = 'Meat Industry',
  DEIMOS_DOWN_PROMO = 'Deimos Down:promo',
  GREAT_DAM_PROMO = 'Great Dam:promo',
  MAGNETIC_FIELD_GENERATORS_PROMO = 'Magnetic Field Generators:promo',
  ASTEROID_RIGHTS = 'Asteroid Rights',
  BIO_PRINTING_FACILITY = 'Bio Printing Facility',
  BACTOVIRAL_RESEARCH = 'Bactoviral Research',
  HARVEST = 'Harvest',
  OUTDOOR_SPORTS = 'Outdoor Sports',
  NEW_PARTNER = 'New Partner',
  MERGER = 'Merger',
  CORPORATE_ARCHIVES = 'Corporate Archives',
  DOUBLE_DOWN = 'Double Down',
  PSYCHE = '16 Psyche',
  ROBOT_POLLINATORS = 'Robot Pollinators',
  HEAD_START = 'Head Start',
  SUPERCAPACITORS = 'Supercapacitors',
  DIRECTED_HEAT_USAGE = 'Directed Heat Usage',

  ANTI_DESERTIFICATION_TECHNIQUES = 'Anti-desertification Techniques',
  AQUEDUCT_SYSTEMS = 'Aqueduct Systems',
  ASTRA_MECHANICA = 'Astra Mechanica',
  CARBON_NANOSYSTEMS = 'Carbon Nanosystems',
  KUIPER_COOPERATIVE = 'Kuiper Cooperative',
  TYCHO_MAGNETICS = 'Tycho Magnetics',
  CYBERIA_SYSTEMS = 'Cyberia Systems',
  ESTABLISHED_METHODS = 'Established Methods',
  GIANT_SOLAR_COLLECTOR = 'Giant Solar Collector',
  HERMETIC_ORDER_OF_MARS = 'Hermetic Order of Mars',
  HOMEOSTASIS_BUREAU = 'Homeostasis Bureau',
  KAGUYA_TECH = 'Kaguya Tech',
  MARS_NOMADS = 'Mars Nomads',
  MARTIAN_LUMBER_CORP = 'Martian Lumber Corp',
  NEPTUNIAN_POWER_CONSULTANTS = 'Neptunian Power Consultants',
  RED_SHIPS = 'Red Ships',
  SOLAR_LOGISTICS = 'Solar Logistics',
  ST_JOSEPH_OF_CUPERTINO_MISSION = 'St. Joseph of Cupertino Mission',
  STRATEGIC_BASE_PLANNING = 'Strategic Base Planning',
  TESLARACT = 'Teslaract',

  ICY_IMPACTORS = 'Icy Impactors',
  SOIL_ENRICHMENT = 'Soil Enrichment',
  CITY_PARKS = 'City Parks',
  SUPERMARKETS = 'Supermarkets',
  HOSPITALS = 'Hospitals',
  PUBLIC_BATHS = 'Public Baths',
  PROTECTED_GROWTH = 'Protected Growth',
  VERMIN = 'Vermin',

  // Promo from contest
  FLOYD_CONTINUUM = 'Floyd Continuum',
  CASINOS = 'Casinos',
  NEW_HOLLAND = 'New Holland',

  // End of promo cards

  // Community corps
  AGRICOLA_INC = 'Agricola Inc',
  CURIOSITY_II = 'Curiosity II',
  INCITE = 'Incite',
  MIDAS = 'Midas',
  PLAYWRIGHTS = 'Playwrights',
  PROJECT_WORKSHOP = 'Project Workshop',
  UNITED_NATIONS_MISSION_ONE = 'United Nations Mission One',
  JUNK_VENTURES = 'Junk Ventures',
  ERIS = 'Eris',
  ATHENA = 'Athena',

  // Community preludes
  VALUABLE_GASES = 'Valuable Gases',
  RESEARCH_GRANT = 'Research Grant',
  AEROSPACE_MISSION = 'Aerospace Mission',
  TRADE_ADVANCE = 'Trade Advance',
  POLITICAL_UPRISING = 'Political Uprising',
  BY_ELECTION = 'By-Election',
  EXECUTIVE_ORDER = 'Executive Order',

  // Community colonies
  SCIENCE_TAG_BLANK_CARD = '',

  // For Playwright.
  SPECIAL_DESIGN_PROXY = 'Special Design:proxy',

  // Ares expansion.
  BIOENGINEERING_ENCLOSURE = 'Bioengineering Enclosure',
  BIOFERTILIZER_FACILITY = 'Bio-Fertilizer Facility',
  BUTTERFLY_EFFECT = 'Butterfly Effect',
  CAPITAL_ARES = 'Capital:ares',
  COMMERCIAL_DISTRICT_ARES = 'Commercial District:ares',
  DEIMOS_DOWN_ARES = 'Deimos Down:ares',
  DESPERATE_MEASURES = 'Desperate Measures',
  ECOLOGICAL_SURVEY = 'Ecological Survey',
  ECOLOGICAL_ZONE_ARES = 'Ecological Zone:ares',
  GEOLOGICAL_SURVEY = 'Geological Survey',
  GREAT_DAM_ARES = 'Great Dam:ares',
  INDUSTRIAL_CENTER_ARES = 'Industrial Center:ares',
  LAVA_FLOWS_ARES = 'Lava Flows:ares',
  MAGNETIC_FIELD_GENERATORS_ARES = 'Magnetic Field Generators:ares',
  MARKETING_EXPERTS = 'Marketing Experts',
  METALLIC_ASTEROID = 'Metallic Asteroid',
  MINING_AREA_ARES = 'Mining Area:ares',
  MINING_RIGHTS_ARES = 'Mining Rights:ares',
  MOHOLE_AREA_ARES = 'Mohole Area:ares',
  NATURAL_PRESERVE_ARES = 'Natural Preserve:ares',
  NUCLEAR_ZONE_ARES = 'Nuclear Zone:ares',
  OCEAN_CITY = 'Ocean City',
  OCEAN_FARM = 'Ocean Farm',
  OCEAN_SANCTUARY = 'Ocean Sanctuary',
  RESTRICTED_AREA_ARES = 'Restricted Area:ares',
  SOLAR_FARM = 'Solar Farm',

  // The Moon.
  MARE_NECTARIS_MINE = 'Mare Nectaris Mine',
  MARE_NUBIUM_MINE = 'Mare Nubium Mine',
  MARE_IMBRIUM_MINE = 'Mare Imbrium Mine',
  MARE_SERENITATIS_MINE = 'Mare Serenitatis Mine',
  HABITAT_14 = 'Habitat 14',
  GEODESIC_TENTS = 'Geodesic Tents',
  SPHERE_HABITATS = 'Sphere Habitats',
  THE_WOMB = 'The Womb',
  TYCHO_ROAD_NETWORK = 'Tycho Road Network',
  ARISTARCHUS_ROAD_NETWORK = 'Aristarchus Road Network',
  SINUS_IRDIUM_ROAD_NETWORK = 'Sinus Irdium Road Network',
  MOMENTUM_VIRUM_HABITAT = 'Momentum Virium Habitat',
  LUNA_TRADE_STATION = 'Luna Trade Station',
  LUNA_MINING_HUB = 'Luna Mining Hub',
  LUNA_TRAIN_STATION = 'Luna Train Station',
  COLONIST_SHUTTLES = 'Colonist Shuttles',
  LUNAR_DUST_PROCESSING_PLANT = 'Lunar Dust Processing Plant',
  DEEP_LUNAR_MINING = 'Deep Lunar Mining',
  ANCIENT_SHIPYARDS = 'Ancient Shipyards',
  LUNA_PROJECT_OFFICE = 'Luna Project Office',
  LUNA_RESORT = 'Luna Resort',
  LUNAR_OBSERVATION_POST = 'Lunar Observation Post',
  MINING_ROBOTS_MANUF_CENTER = 'Mining Robots Manuf. Center',
  PRIDE_OF_THE_EARTH_ARKSHIP = 'Pride of the Earth Arkship',
  IRON_EXTRACTION_CENTER = 'Iron Extraction Center',
  TITANIUM_EXTRACTION_CENTER = 'Titanium Extraction Center',
  ARCHIMEDES_HYDROPONICS_STATION = 'Archimedes Hydroponics Station',
  STEEL_MARKET_MONOPOLISTS = 'Steel Market Monopolists',
  TITANIUM_MARKET_MONOPOLISTS = 'Titanium Market Monopolists',
  LUNA_STAGING_STATION = 'Luna Staging Station',
  NEW_COLONY_PLANNING_INITIAITIVES = 'New Colony Planning Initiatives',
  AI_CONTROLLED_MINE_NETWORK = 'AI Controlled Mine Network',
  DARKSIDE_METEOR_BOMBARDMENT = 'Darkside Meteor Bombardment',
  UNDERGROUND_DETONATORS = 'Underground Detonators',
  LUNAR_TRADE_FLEET = 'Lunar Trade Fleet',
  SUBTERRANEAN_HABITATS = 'Subterranean Habitats',
  IMPROVED_MOON_CONCRETE = 'Improved Moon Concrete',
  MOONCRATE_BLOCK_FACTORY = 'Mooncrate Block Factory',
  HEAVY_DUTY_ROVERS = 'Heavy Duty Rovers',
  MICROSINGULARITY_PLANT = 'Microsingularity Plant',
  HELIOSTAT_MIRROR_ARRAY = 'Heliostat Mirror Array',
  LUNAR_SECURITY_STATIONS = 'Lunar Security Stations',
  HYPERSENSITIVE_SILICON_CHIP_FACTORY = 'Hypersensitive Silicon Chip Factory',
  COPERNICUS_SOLAR_ARRAYS = 'Copernicus Solar Arrays',
  DARKSIDE_INCUBATION_PLANT = 'Darkside Incubation Plant',
  WATER_TREATMENT_COMPLEX = 'Water Treatment Complex',
  ALGAE_BIOREACTORS = 'Algae Bioreactors',
  HE3_FUSION_PLANT = 'HE3 Fusion Plant',
  HE3_REFINERY = 'HE3 Refinery',
  HE3_LOBBYISTS = 'HE3 Lobbyists',
  REVOLTING_COLONISTS = 'Revolting Colonists',
  COSMIC_RADIATION = 'Cosmic Radiation',
  OFF_WORLD_CITY_LIVING = 'Off-World City Living',
  ROAD_PIRACY = 'Road Piracy',
  LUNAR_MINE_URBANIZATION = 'Lunar Mine Urbanization',
  THORIUM_RUSH = 'Thorium Rush',
  HE3_PRODUCTION_QUOTAS = 'HE3 Production Quotas',
  LUNA_CONFERENCE = 'Luna Conference',
  WE_GROW_AS_ONE = 'We Grow As One',
  MOONCRATE_CONVOYS_TO_MARS = 'Mooncrate Convoys To Mars',
  LUNAR_INDEPENDENCE_WAR = 'Lunar Independence War',
  AN_OFFER_YOU_CANT_REFUSE = 'An Offer You Can\'t Refuse',
  PRELIMINARY_DARKSIDE = 'Preliminary Darkside',
  HOSTILE_TAKEOVER = 'Hostile Takeover',
  SYNDICATE_PIRATE_RAIDS = 'Syndicate Pirate Raids',
  DARKSIDE_MINING_SYNDICATE = 'Darkside Mining Syndicate',
  HE3_PROPULSION = 'HE3 Propulsion',
  STAGING_STATION_BEHEMOTH = 'Staging Station "Behemoth"',
  LUNA_ARCHIVES = 'Luna Archives',
  LUNA_SENATE = 'Luna Senate',
  LUNA_POLITICAL_INSTITUTE = 'Luna Political Institute',
  COPERNICUS_TOWER = 'Copernicus Tower',
  SMALL_DUTY_ROVERS = 'Small Duty Rovers',
  LUNAR_INDUSTRY_COMPLEX = 'Lunar Industry Complex',
  DARKSIDE_OBSERVATORY = 'Darkside Observatory',
  MARTIAN_EMBASSY = 'Martian Embassy',
  EARTH_EMBASSY = 'Earth Embassy',
  ROVER_DRIVERS_UNION = 'Rover Drivers Union',
  LTF_HEADQUARTERS = 'L.T.F. Headquarters',
  DARKSIDE_SMUGGLERS_UNION = 'Darkside Smugglers\' Union',
  UNDERMOON_DRUG_LORDS_NETWORK = 'Undermoon Drug Lords Network',
  LTF_PRIVILEGES = 'L.T.F. Privileges',
  GRAND_LUNA_ACADEMY = 'Grand Luna Academy',
  LUNA_ECUMENOPOLIS = 'Luna Ecumenopolis',
  ORBITAL_POWER_GRID = 'Orbital Power Grid',
  PROCESSOR_FACTORY = 'Processor Factory',
  LUNAR_STEEL = 'Lunar Steel',
  RUST_EATING_BACTERIA = 'Rust Eating Bacteria',
  SOLAR_PANEL_FOUNDRY = 'Solar Panel Foundry',
  MOON_TETHER = 'Moon Tether',
  NANOTECH_INDUSTRIES = 'Nanotech Industries',
  TEMPEST_CONSULTANCY = 'Tempest Consultancy',
  THE_DARKSIDE_OF_THE_MOON_SYNDICATE = 'The Darkside of The Moon Syndicate',
  LUNA_HYPERLOOP_CORPORATION = 'Luna Hyperloop Corporation',
  CRESCENT_RESEARCH_ASSOCIATION = 'Crescent Research Association',
  LUNA_FIRST_INCORPORATED = 'Luna First Incorporated',
  THE_GRAND_LUNA_CAPITAL_GROUP = 'The Grand Luna Capital Group',
  INTRAGEN_SANCTUARY_HEADQUARTERS = 'Intragen Sanctuary Headquarters',
  LUNA_TRADE_FEDERATION = 'Luna Trade Federation',
  THE_ARCHAIC_FOUNDATION_INSTITUTE = 'The Archaic Foundation Institute',
  FIRST_LUNAR_SETTLEMENT = 'First Lunar Settlement',
  CORE_MINE = 'Core Mine',
  BASIC_INFRASTRUCTURE = 'Basic Infrastructure',
  LUNAR_PlANNING_OFFICE = 'Lunar Planning Office',
  MINING_COMPLEX = 'Mining Complex',
  MOON_ROAD_STANDARD_PROJECT = 'Road Infrastructure',
  MOON_MINE_STANDARD_PROJECT = 'Lunar Mine',
  MOON_HABITAT_STANDARD_PROJECT = 'Lunar Habitat',
  MOON_ROAD_STANDARD_PROJECT_VARIANT_1 = 'Road Infrastructure (var. 1)',
  MOON_MINE_STANDARD_PROJECT_VARIANT_1 = 'Lunar Mine (var. 1)',
  MOON_HABITAT_STANDARD_PROJECT_VARIANT_1 = 'Lunar Habitat (var. 1)',
  MOON_ROAD_STANDARD_PROJECT_VARIANT_2 = 'Road Infrastructure (var. 2)',
  MOON_MINE_STANDARD_PROJECT_VARIANT_2 = 'Lunar Mine (var. 2)',
  MOON_HABITAT_STANDARD_PROJECT_VARIANT_2 = 'Lunar Habitat (var. 2)',

  // Pathfinders
  BREEDING_FARMS = 'Breeding Farms',
  PREFABRICATION_OF_HUMAN_HABITATS = 'Prefabrication of Human Habitats',
  NEW_VENICE = 'New Venice',
  AGRO_DRONES = 'Agro-Drones',
  WETLANDS = 'Wetlands',
  RARE_EARTH_ELEMENTS = 'Rare-Earth Elements',
  ORBITAL_LABORATORIES = 'Orbital Laboratories',
  DUST_STORM = 'Dust Storm',
  MARTIAN_MONUMENTS = 'Martian Monuments',
  MARTIAN_NATURE_WONDERS = 'Martian Nature Wonders',
  MUSEUM_OF_EARLY_COLONISATION = 'Museum of Early Colonisation',
  TERRAFORMING_CONTROL_STATION = 'Terraforming Control Station',
  MARTIAN_TRANSHIPMENT_STATION = 'Martian Transhipment Station',
  CERES_SPACEPORT = 'Ceres Spaceport',
  DYSON_SCREENS = 'Dyson Screens',
  LUNAR_EMBASSY = 'Lunar Embassy',
  GEOLOGICAL_EXPEDITION = 'Geological Expedition',
  EARLY_EXPEDITION = 'Early Expedition',
  HYDROGEN_PROCESSING_PLANT = 'Hydrogen Processing Plant',
  POWER_PLANT_PATHFINDERS = 'Power Plant:Pathfinders',
  LUXURY_ESTATE = 'Luxury Estate',
  RETURN_TO_ABANDONED_TECHNOLOGY = 'Return to Abandoned Technology',
  DESIGNED_ORGANISMS = 'Designed Organisms',
  SPACE_DEBRIS_CLEANING_OPERATION = 'Space Debris Cleaning Operation',
  PRIVATE_SECURITY = 'Private Security',
  SECRET_LABS = 'Secret Labs',
  CYANOBACTERIA = 'Cyanobacteria',
  COMMUNICATION_CENTER = 'Communication Center',
  MARTIAN_REPOSITORY = 'Martian Repository',
  DATA_LEAK = 'Data Leak',
  SMALL_OPEN_PIT_MINE = 'Small Open Pit Mine',
  SOLAR_STORM = 'Solar Storm',
  SPACE_RELAY = 'Space Relay',
  DECLARATION_OF_INDEPENDENCE = 'Declaration of Independence',
  MARTIAN_CULTURE = 'Martian Culture',
  OZONE_GENERATORS = 'Ozone Generators',
  SMALL_COMET = 'Small Comet',
  ECONOMIC_ESPIONAGE = 'Economic Espionage',
  FLAT_MARS_THEORY = 'Flat Mars Theory',
  ASTEROID_RESOURCES = 'Asteroid Resources',
  KICKSTARTER = 'Kickstarter',
  ECONOMIC_HELP = 'Economic Help',
  INTERPLANETARY_TRANSPORT = 'Interplanetary Transport',
  MARTIAN_DUST_PROCESSING_PLANT = 'Martian Dust Processing Plant',
  CULTIVATION_OF_VENUS = 'Cultivation of Venus',
  EXPEDITION_TO_THE_SURFACE_VENUS = 'Expedition to the Surface - Venus',
  LAST_RESORT_INGENUITY = 'Last Resort Ingenuity',
  CRASHLANDING = 'Crashlanding',
  THINK_TANK = 'Think Tank',
  BOTANICAL_EXPERIENCE = 'Botanical Experience',
  CRYPTOCURRENCY = 'Cryptocurrency',
  RICH_DEPOSITS = 'Rich Deposits',
  OUMUAMUA_TYPE_OBJECT_SURVEY = 'Oumuamua Type Object Survey',
  SOLARPEDIA = 'Solarpedia',
  ANTHOZOA = 'Anthozoa',
  ADVANCED_POWER_GRID = 'Advanced Power Grid',
  SPECIALIZED_SETTLEMENT = 'Specialized Settlement',
  CHARITY_DONATION = 'Charity Donation',
  CURIOSITY_LABS = 'Curiosity Labs',
  NOBEL_LABS = 'Nobel Labs',
  HUYGENS_OBSERVATORY = 'Huygens Observatory',
  CASSINI_STATION = 'Cassini Station',
  MICROBIOLOGY_PATENTS = 'Microbiology Patents',
  COORDINATED_RAID = 'Coordinated Raid',
  LOBBY_HALLS = 'Lobby Halls',
  RED_CITY = 'Red City',
  VENERA_BASE = 'Venera Base',
  GATEWAY_STATION = 'Gateway Station',
  FLOATER_URBANISM = 'Floater-Urbanism',
  SOIL_DETOXIFICATION = 'Soil Detoxification',
  HIGH_TEMP_SUPERCONDUCTORS = 'High Temp. Superconductors',
  PUBLIC_SPONSORED_GRANT = 'Public Sponsored Grant',
  POLLINATORS = 'Pollinators',
  SOCIAL_EVENTS = 'Social Events',
  CONTROLLED_BLOOM = 'Controlled Bloom',
  TERRAFORMING_ROBOTS = 'Terraforming Robots',

  VENUS_FIRST = 'Venus First',
  VALUABLE_GASES_PATHFINDERS = 'Valuable Gases:Pathfinders',
  CO2_REDUCERS = 'CO² Reducers',
  HYDROGEN_BOMBARDMENT = 'Hydrogen Bombardment',
  RESEARCH_GRANT_PATHFINDERS = 'Research Grant:Pathfinders',
  CREW_TRAINING = 'Crew Training',
  SURVEY_MISSION = 'Survey Mission',
  DESIGN_COMPANY = 'Design Company',
  CONSOLIDATION = 'Consolidation',
  PERSONAL_AGENDA = 'Personal Agenda',
  VITAL_COLONY = 'Vital Colony',
  DEEP_SPACE_OPERATIONS = 'Deep Space Operations',
  EXPERIENCED_MARTIANS = 'Experienced Martians',
  THE_NEW_SPACE_RACE = 'The New Space Race',

  POLARIS = 'Polaris',
  PLANET_PR = 'planet pr',
  AMBIENT = 'Ambient',
  RINGCOM = 'Ringcom',
  CHIMERA = 'Chimera',
  SISTEMAS_SEEBECK = 'Sistemas Seebeck',
  // SPIRE = 'Spire',
  SOYLENT_SEEDLING_SYSTEMS = 'Soylent Seedling Systems',
  STEELARIS = 'Steelaris',
  MARS_MATHS = 'Mars Maths',
  MARS_DIRECT = 'Mars Direct',
  MARTIAN_INSURANCE_GROUP = 'Martian Insurance Group',
  SOLBANK = 'SolBank',
  BIO_SOL = 'Bio-Sol',
  AURORAI = 'Aurorai',
  COLLEGIUM_COPERNICUS = 'Collegium Copernicus',
  ROBIN_HAULINGS = 'Robin Haulings',
  ODYSSEY = 'Odyssey',
  GAGARIN_MOBILE_BASE = 'Gagarin Mobile Base',
  MARS_FRONTIER_ALLIANCE = 'Mars Frontier Alliance',
  MIND_SET_MARS = 'Mind Set Mars',
  HABITAT_MARTE = 'Habitat Marte',
  ADHAI_HIGH_ORBIT_CONSTRUCTIONS = 'Adhai High Orbit Constructions',

  // CEOs
  ASIMOV = 'Asimov',
  BJORN = 'Bjorn',
  CLARKE = 'Clarke',
  DUNCAN = 'Duncan',
  ENDER = 'Ender',
  FLOYD = 'Floyd',
  GORDON = 'Gordon',
  GRETA = 'Greta',
  HAL9000 = 'HAL 9000',
  INGRID = 'Ingrid',
  JANSSON = 'Jansson',
  KAREN = 'Karen',
  LOWELL = 'Lowell',
  MUSK = 'Musk',
  MARIA = 'Maria',
  NAOMI = 'Naomi',
  OSCAR = 'Oscar',
  PETRA = 'Petra',
  QUILL = 'Quill',
  ROGERS = 'Rogers',
  STEFAN = 'Stefan',
  TATE = 'Tate',
  ULRICH = 'Ulrich',
  VANALLEN = 'Van Allen',
  WILL = 'Will',
  XAVIER = 'Xavier',
  YVONNE = 'Yvonne',
  ZAN = 'Zan',
  APOLLO = 'Apollo',
  CAESAR = 'Caesar',
  FARADAY = 'Faraday',
  GAIA = 'Gaia',
  HUAN = 'Huan',
  NEIL = 'Neil',
  RYU = 'Ryu',
  SHARA = 'Shara',
  XU = 'Xu',
  // CEO Preludes
  CO_LEADERSHIP = 'Co-leadership',

  // Star Wars cards
  TRADE_EMBARGO = 'Trade Embargo (I)',
  CLONE_TROOPERS = 'Clone Troopers (II)',
  BEHOLD_THE_EMPEROR = 'Behold The Emperor! (III)',
  TOSCHE_STATION = 'Tosche Station (IV)',
  CLOUD_CITY = 'Cloud City (V)',
  FOREST_MOON = 'Forest Moon (VI)',
  TAKONDA_CASTLE = 'Takonda Castle (VII)',
  TOOL_WITH_THE_FIRST_ORDER = 'Tool with the First Order (VIII)',
  REY_SKYWALKER = 'Rey ... Skywalker?! (IX)',

  // Underworld
  GEOLOGIST_TEAM = 'Geologist Team',
  GEOSCAN_SATELLITE = 'Geoscan Satellite',
  TUNNEL_BORING_MACHINE = 'Tunnel Boring Machine',
  UNDERGROUND_RAILWAY = 'Underground Railway',
  GAIA_CITY = 'Gaia City',
  DEEPNUKING = 'Deepnuking',
  OLD_WORLD_MAFIA = 'Old World Mafia',
  NIGHTCLUBS = 'Nightclubs',
  RECKLESS_DETONATION = 'Reckless Detonation',
  OFF_WORLD_TAX_HAVEN = 'Off-World Tax Haven',
  SUBNAUTIC_PIRATES = 'Subnautic Pirates',
  SOCIAL_ENGINEERING = 'Social Engineering',
  FABRICATED_SCANDAL = 'Fabricated Scandal',
  LABOR_TRAFFICKING = 'Labor Trafficking',
  SUBTERRANEAN_SEA = 'Subterranean Sea',
  FOREST_TUNNELS = 'Forest Tunnels',
  MAN_MADE_VOLCANO = 'Man-made Volcano',
  TUNNELING_SUBCONTRACTOR = 'Tunneling Subcontractor',
  UNDERGROUND_AMUSEMENT_PARK = 'Underground Amusement Park',
  CASINO = 'Casino',
  IMPORTED_HEAVY_MACHINERY = 'Imported Heavy Machinery',
  MICROPROBING_TECHNOLOGY = 'Microprobing Technology',
  SEARCH_FOR_LIFE_UNDERGROUND = 'Search for Life Underground',
  GEOTHERMAL_NETWORK = 'Geothermal Network',
  GLOBAL_AUDIT = 'Global Audit',
  PATENT_MANIPULATION = 'Patent Manipulation',
  CAVE_CITY = 'Cave City',
  UNDERGROUND_SMUGGLING_RING = 'Underground Smuggling Ring',
  DEEPMINING = 'Deepmining',
  BEHEMOTH_EXCAVATOR = 'Behemoth Excavator',
  LOBBYING_NETWORK = 'Lobbying Network',
  CONCESSION_RIGHTS = 'Concession Rights',
  ORBITAL_LASER_DRILL = 'Orbital Laser Drill',
  GREY_MARKET_EXPLOITATION = 'Grey Market Exploitation',
  EXCAVATOR_LEASING = 'Excavator Leasing',
  LANDFILL = 'Landfill',
  NARRATIVE_SPIN = 'Narrative Spin',
  PRIVATE_INVESTIGATOR = 'Private Investigator',
  CORPORATE_BLACKMAIL = 'Corporate Blackmail',
  SCAPEGOAT = 'Scapegoat',
  FRIENDS_IN_HIGH_PLACES = 'Friends in High Places',
  MICROGRAVIMETRY = 'Microgravimetry',
  STEM_FIELD_SUBSIDIES = 'Stem Field Subsidies',
  TITAN_MANUFACTURING_COLONY = 'Titan Manufacturing Colony',
  ROBOT_MOLES = 'Robot Moles',
  MINING_MARKET_INSIDER = 'Mining Market Insider',
  SERVER_SABOTAGE = 'Server Sabotage',
  SPACE_WARGAMES = 'Space Wargames',
  PRIVATE_MILITARY_CONTRACTOR = 'Private Military Contractor',
  SPACE_PRIVATEERS = 'Space Privateers',
  PERSONAL_SPACECRUISER = 'Personal Spacecruiser',
  HYPERSPACE_DRIVE_PROTOTYPE = 'Hyperspace Drive Prototype',
  STAR_VEGAS = 'Star Vegas',
  PRIVATE_RESORTS = 'Private Resorts',
  EARTHQUAKE_MACHINE = 'Earthquake Machine',
  MICRO_GEODESICS = 'Micro-Geodesics',
  NEUTRINOGRAPH = 'Neutrinograph',
  SOIL_EXPORT = 'Soil Export',
  ARTESIAN_AQUIFER = 'Artesian Aquifer',
  CHEMICAL_FACTORY = 'Chemical Factory',
  CORPORATE_THEFT = 'Corporate Theft',
  UNDERGROUND_RESEARCH_CENTER = 'Underground Research Center',
  PRICE_WARS = 'Price Wars',
  ANTI_TRUST_CRACKDOWN = 'Anti-trust Crackdown',
  MONOPOLY = 'Monopoly',
  STAGED_PROTESTS = 'Staged Protests',
  PLANT_TAX = 'Plant Tax',
  INFRASTRUCTURE_OVERLOAD = 'Infrastructure Overload',
  CRATER_SURVEY = 'Crater Survey',
  INDUCED_TREMOR = 'Induced Tremor',
  UNDERGROUND_HABITAT = 'Underground Habitat',
  UNDERGROUND_SHELTERS = 'Underground Shelters',
  VOLUNTEER_MINING_INITIATIVE = 'Volunteer Mining Initiative',
  NANOFOUNDRY = 'Nanofoundry',
  BATTERY_FACTORY = 'Battery Factory',
  VOLTAIC_METALLURGY = 'Voltaic Metallurgy',
  PUBLIC_SPACELINE = 'Public Spaceline',
  MARTIAN_EXPRESS = 'Martian Express',
  EXPEDITION_VEHICLES = 'Expedition Vehicles',
  CUT_THROAT_BUDGETING = 'Cut-throat Budgeting',
  GEOLOGICAL_SURVEY_UNDERWORLD = 'Geological Survey:underworld',
  CLASS_ACTION_LAWSUIT = 'Class-action Lawsuit',
  MERCENARY_SQUAD = 'Mercenary Squad',
  RESEARCH_DEVELOPMENT_HUB = 'Research & Development Hub',
  PLANETARY_RIGHTS_BUYOUT = 'Planetary Rights Buyout',
  MEDIA_FRENZY = 'Media Frenzy',
  INVESTIGATIVE_JOURNALISM = 'Investigative Journalism',
  WHALES = 'Whales',
  GUERILLA_ECOLOGISTS = 'Guerilla Ecologists',
  THIOLAVA_VENTS = 'Thiolava Vents',
  DETECTIVE_TV_SERIES = 'Detective TV Series',
  RACKETEERING = 'Racketeering',
  GAS_TRUST = 'Gas Trust',
  STING_OPERATION = 'Sting Operation',
  FAMILY_CONNECTIONS = 'Family Connections',
  BIOBATTERIES = 'Biobatteries',
  EXPORT_CONVOY = 'Export Convoy',
  ACIDIZING = 'Acidizing',
  EXPLOITATION_OF_VENUS = 'Exploitation Of Venus',

  // Underworld Corporations
  HADESPHERE = 'Hadesphere',
  DEMETRON_LABS = 'Demetron Labs',
  JENSON_BOYLE_CO = 'Jenson-Boyle & Co',
  HENKEI_GENETICS = 'Henkei Genetics',
  ARBORIST_COLLECTIVE = 'Arborist Collective',
  KINGDOM_OF_TAURARO = 'Kingdom of Tauraro',
  AERON_GENOMICS = 'Aeron Genomics',
  KEPLERTEC = 'Keplertec',
  VOLTAGON = 'Voltagon',
  ARES_MEDIA = 'Ares Media',
  ANUBIS_SECURITIES = 'Anubis Securities',
  HECATE_SPEDITIONS = 'Hecate Speditions',
  // Underworld Preludes
  FREE_TRADE_PORT = 'Free Trade Port',
  INVESTOR_PLAZA = 'Investor Plaza',
  INHERITED_FORTUNE = 'Inherited Fortune',
  INTELLECTUAL_PROPERTY_THEFT = 'Intellectual Property Theft',
  TUNNELING_OPERATION = 'Tunneling Operation',
  GEOLOGICAL_EXPERTISE = 'Geological Expertise',
  UNDERGROUND_SETTLEMENT = 'Underground Settlement',
  GANYMEDE_TRADING_COMPANY = 'Ganymede Trading Company',
  CENTRAL_RESERVOIR = 'Central Reservoir',
  BATTERY_SHIPMENT = 'Battery Shipment',
  DEEPWATER_DOME = 'Deepwater Dome',
  SECRET_RESEARCH = 'Secret Research',
  PROSPECTING = 'Prospecting',
  ELECTION_SPONSORSHIP = 'Election Sponsorship',
  CLOUD_VORTEX_OUTPOST = 'Cloud Vortex Outpost',

  // Underworld Standard Projects
  EXCAVATE_STANDARD_PROJECT = 'Excavate:SP',
  COLLUSION_STANDARD_PROJECT = 'Collusion:SP',

  // Underworld Replacement Cards
  STANDARD_TECHNOLOGY_UNDERWORLD = 'Standard Technology:u',
  HACKERS_UNDERWORLD = 'Hackers:u',
  HIRED_RAIDERS_UNDERWORLD = 'Hired Raiders:u',
}


export enum BoardName {
  THARSIS = 'tharsis',
  HELLAS = 'hellas',
  ELYSIUM = 'elysium',

  UTOPIA_PLANITIA = 'utopia planitia',
  VASTITAS_BOREALIS_NOVUS = 'vastitas borealis novus',
  TERRA_CIMMERIA_NOVUS = 'terra cimmeria novus',

  ARABIA_TERRA = 'arabia terra',
  VASTITAS_BOREALIS = 'vastitas borealis',
  AMAZONIS = 'amazonis p.',
  TERRA_CIMMERIA = 't. cimmeria',
}

export type AgendaStyle =
  'Standard' |
  'Random' |
  'Chairman';

const PARTIES = ['m', 's', 'u', 'k', 'r', 'g'] as const;
const BONUS_SUFFIXES = ['b01', 'b02'] as const;
const POLICY_SUFFIXES = ['p01', 'p02', 'p03', 'p04'] as const;

type Party = typeof PARTIES[number];
type BonusSuffix = typeof BONUS_SUFFIXES[number]
type PolicySuffix = typeof POLICY_SUFFIXES[number];

export type BonusId = `${Party}${BonusSuffix}`;
export type PolicyId = `${Party}${PolicySuffix}`

export type Agenda = {
  bonusId: BonusId;
  policyId: PolicyId;
}


export enum RandomMAOptionType {
    NONE = 'No randomization',
    LIMITED = 'Limited synergy',
    UNLIMITED = 'Full random'
}


export type GameOptionsModel = {
  aresExtremeVariant: boolean,
  altVenusBoard: boolean,
  boardName: BoardName,
  bannedCards: ReadonlyArray<CardName>;
  expansions: Record<Expansion, boolean>,
  draftVariant: boolean,
  escapeVelocityMode: boolean,
  escapeVelocityThreshold?: number,
  escapeVelocityBonusSeconds?: number,
  escapeVelocityPeriod?: number,
  escapeVelocityPenalty?: number,
  fastModeOption: boolean,
  includedCards: ReadonlyArray<CardName>;
  includeFanMA: boolean,
  initialDraftVariant: boolean,
  preludeDraftVariant: boolean,
  politicalAgendasExtension: AgendaStyle,
  removeNegativeGlobalEvents: boolean,
  showOtherPlayersVP: boolean,
  showTimers: boolean,
  shuffleMapOption: boolean,
  solarPhaseOption: boolean,
  soloTR: boolean,
  randomMA: RandomMAOptionType,
  requiresMoonTrackCompletion: boolean,
  requiresVenusTrackCompletion: boolean,
  twoCorpsVariant: boolean,
  undoOption: boolean,
}


export enum ColonyName {
    CALLISTO = 'Callisto',
    CERES = 'Ceres',
    ENCELADUS = 'Enceladus',
    EUROPA = 'Europa',
    GANYMEDE = 'Ganymede',
    IO = 'Io',
    LUNA = 'Luna',
    MIRANDA = 'Miranda',
    PLUTO = 'Pluto',
    TITAN = 'Titan',
    TRITON = 'Triton',

    // Community
    // If you add a community colony, update
    // ColonyDealer.includesCommunityColonies
    IAPETUS = 'Iapetus',
    MERCURY = 'Mercury',
    HYGIEA = 'Hygiea',
    TITANIA = 'Titania',
    VENUS = 'Venus',
    LEAVITT = 'Leavitt',
    PALLAS = 'Pallas',
    DEIMOS = 'Deimos',

    // Pathfinders
    LEAVITT_II = 'Leavitt II',
    IAPETUS_II = 'Iapetus II',

    // WHEN ADDING A NEW COLONY, ADD IT TO AllColonies.ts
}


export type ColonyModel = {
  colonies: Array<Color>;
  isActive: boolean;
  name: ColonyName;
  trackPosition: number;
  visitor: Color | undefined;
}


export type PlayerId = `p${string}`;
export type GameId = `g${string}`;
export type SpectatorId = `s${string}`;
export type ParticipantId = PlayerId | SpectatorId;
type Digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9';
type TwoDigits = `${Digit}${Digit}`;
export type SpaceId = `${TwoDigits}` | `m${TwoDigits}`;
export type Named<T> = {name: T};

export type MilestoneCount = {
  id: PlayerId;
  // TODO(kberg): Remove count by 2025-08-01
  count?: number;
  networkerCount: number;
  purifierCount: number;
}

export const PLAYER_COLORS = ['red', 'green', 'yellow', 'blue', 'black', 'purple', 'orange', 'pink'] as const;
const ALL_COLORS = [...PLAYER_COLORS, 'neutral', 'bronze'] as const;
export type Color = typeof ALL_COLORS[number];
export type ColorWithNeutral = Color | 'NEUTRAL';

export type AwardScore = {
  playerColor: Color;
  playerScore: number;
}

export const awardNames = [
  // Tharsis
  'Landlord',
  'Scientist',
  'Banker',
  'Thermalist',
  'Miner',

  // Elysium
  'Celebrity',
  'Industrialist',
  'Desert Settler',
  'Estate Dealer',
  'Benefactor',

  // Hellas
  'Contractor',
  'Cultivator',
  'Excentric',
  'Magnate',
  'Space Baron',

  // Venus
  'Venuphile',

  // Ares
  'Entrepreneur',

  // The Moon
  'Full Moon',
  'Lunar Magnate',

  // Amazonis Planitia
  // NB: the fifth award for Amazonis Plantia is Promoter, also part of Arabia Terra.
  'Curator',
  'A. Engineer',
  'Tourist',
  'A. Zoologist',

  // Arabia Terra
  'Cosmic Settler',
  'Botanist',
  'Promoter',
  'A. Manufacturer',
  'Zoologist',

  // Terra Cimmeria
  'Biologist',
  'T. Politician',
  'Urbanist',
  'Warmonger',
  // NB: the fifth award for Terra Cimmeria is Incorporator, a modular award.

  // Vastitas Borealis
  'Forecaster',
  'Edgedancer',
  'Visionary',
  'Naturalist',
  'Voyager',

  // Vastitas Borealis Novus
  'Traveller',
  'Landscaper',
  'Highlander',
  'Manufacturer',

  // Underworld
  'Kingpin',
  'EdgeLord',

  // Ares Extreme
  'Rugged',

  // Modular
  'Administrator',
  'Collector',
  'Constructor',
  'Electrician',
  'Founder',
  'Incorporator',
  'Investor',
  'Metropolist',
  'Mogul',
  'Politician', // New Most party leaders and influence compbined
  // 'Suburbian', // NEW Most tiles on areas along the edges of the map.
  // 'Zoologist', // Most animal and microbe resources. Currently Zoologist2
] as const;

export type AwardName = typeof awardNames[number];

export type FundedAwardModel = {
  name: AwardName;
  playerName: string | undefined;
  playerColor: Color | undefined;
  scores: Array<AwardScore>;
}


export const HAZARD_CONSTRAINTS = [
  'erosionOceanCount',
  'removeDustStormsOceanCount',
  'severeErosionTemperature',
  'severeDustStormOxygen',
] as const;

export type HazardConstraint = {
    threshold: number,
    available: boolean
}

export type HazardData = Record<typeof HAZARD_CONSTRAINTS[number], HazardConstraint>;

export type AresData = {
  includeHazards: boolean;
  hazardData: HazardData;
  milestoneResults: Array<MilestoneCount>;
}

export enum GlobalParameter {
  OCEANS = 'oceans',
  OXYGEN = 'oxygen',
  TEMPERATURE = 'temperature',
  VENUS = 'venus',
  MOON_HABITAT_RATE = 'moon-habitat',
  MOON_MINING_RATE = 'moon-mining',
  MOON_LOGISTICS_RATE = 'moon-logistics',
}

export const GLOBAL_PARAMETERS = [
  GlobalParameter.OCEANS,
  GlobalParameter.OXYGEN,
  GlobalParameter.TEMPERATURE,
  GlobalParameter.VENUS,
  GlobalParameter.MOON_HABITAT_RATE,
  GlobalParameter.MOON_MINING_RATE,
  GlobalParameter.MOON_LOGISTICS_RATE,
] as const;

export type MilestoneScore = {
  playerColor: Color;
  playerScore: number;
}

export const milestoneNames = [
  // Tharsis
  'Terraformer',
  'Mayor',
  'Gardener',
  'Planner',
  'Builder',

  // Elysium
  'Generalist',
  'Specialist',
  'Ecologist',
  'Tycoon',
  'Legend',

  // Hellas
  'Diversifier',
  'Tactician',
  'Polar Explorer',
  'Energizer',
  'Rim Settler',

  // Venus
  'Hoverlord',

  // Ares
  'Networker',

  // The Moon
  'One Giant Step',
  'Lunarchitect',

  // Amazonis Planitia
  'Colonizer',
  'Minimalist',
  'Terran',
  'Tropicalist',

  // Arabia Terra
  'Economizer',
  'Pioneer',
  'Land Specialist',
  'Martian',

  // Terra Cimmeria
  'T. Collector',
  'Firestarter',
  'Terra Pioneer',
  'Spacefarer', // TODO(kberg): Rename to T. Spacefarer
  'Gambler',

  // Vastitas Borealis
  'V. Electrician',
  'Smith',
  'Tradesman',
  'Irrigator',
  'Capitalist',

  // Vastitas Borealis Novus
  'Agronomist',
  'Engineer',
  'V. Spacefarer',
  'Geologist',
  'Farmer',

  // Underworld
  'Tunneler',
  'Risktaker',

  // Ares Extreme
  'Purifier',

  // Modular
  // 'Briber',
  // 'Builder', // But 7 building tags
  // 'Coastguard', // NEW 3 tiles adjacent to oceans
  // 'Farmer',
  'Forester',
  'Fundraiser',
  'Hydrologist',
  'Landshaper',
  // 'Legend', // But 4 events
  'Lobbyist',
  // 'Merchant',
  // 'Metallurgist', // Smith, but 6
  'Philantropist',
  // 'Pioneer', // But 4 colonies
  'Planetologist',
  'Producer',
  'Researcher',
  // 'Spacefarer', // But 4 space tags
  'Sponsor',
  // 'Tactician', // but 4 cards with requirements
  // 'Terraformer', // but 29 TR
  // 'Terran', // But 5 Earth tags.
  'Thawer',
  // 'Trader', // NEW 3 types of resources on cards.
  // 'Tycoon', // But, 10 Green and Blue cards combined.
] as const;

export type MilestoneName = typeof milestoneNames[number];

export type ClaimedMilestoneModel = {
  name: MilestoneName;
  playerName: string | undefined;
  playerColor: Color | undefined;
  scores: Array<MilestoneScore>;
}

export type SpaceHighlight = undefined | 'noctis' | 'volcanic';

export enum SpaceType {
    LAND = 'land',
    OCEAN = 'ocean',
    COLONY = 'colony',
    LUNAR_MINE = 'lunar_mine', // Reserved for The Moon.
    COVE = 'cove', // Cove can represent an ocean and a land space.
    RESTRICTED = 'restricted', // Amazonis Planitia
}

export enum SpaceBonus {
    TITANIUM, // 0
    STEEL, // 1
    PLANT, // 2
    DRAW_CARD, // 3
    HEAT, // 4
    OCEAN, // 5

    // Ares-specific
    MEGACREDITS, // 6
    ANIMAL, // 7 (Also used in Amazonis)
    MICROBE, // 8 (Also used in Arabia Terra)
    ENERGY, // 9 // Ares and Terra Cimmeria

    // Arabia Terra-specific
    DATA, // 10
    SCIENCE, // 11
    ENERGY_PRODUCTION, // 12

    // Vastitas Borealis-specific (and Terra Cimmeria)
    TEMPERATURE, // 13

    // Amazonis-specific
    _RESTRICTED, // 14
    ASTEROID, // 15 // Used by Deimos Down Ares

    // Vastitas Borealis Novus-specific
    DELEGATE, // 16
    // Terra Cimmeria Novus-specific
    COLONY, // 17

}


// for now.
export enum TileType {
    GREENERY, // 0
    OCEAN, // 1
    CITY, // 2
    CAPITAL, // 3
    COMMERCIAL_DISTRICT, // 4
    ECOLOGICAL_ZONE, // 5
    INDUSTRIAL_CENTER, // 6
    LAVA_FLOWS, // 7
    MINING_AREA, // 8
    MINING_RIGHTS, // 9
    MOHOLE_AREA, // 10
    NATURAL_PRESERVE, // 11
    NUCLEAR_ZONE, // 12
    RESTRICTED_AREA, // 13

    DEIMOS_DOWN, // 14
    GREAT_DAM, // 15
    MAGNETIC_FIELD_GENERATORS, // 16

    BIOFERTILIZER_FACILITY, // 17
    METALLIC_ASTEROID, // 18
    SOLAR_FARM, // 19
    OCEAN_CITY, // 20, Also used in Pathfinders
    OCEAN_FARM, // 21
    OCEAN_SANCTUARY, // 22
    DUST_STORM_MILD, // 23
    DUST_STORM_SEVERE, // 24
    EROSION_MILD, // 25
    EROSION_SEVERE, // 26
    MINING_STEEL_BONUS, // 27
    MINING_TITANIUM_BONUS, // 28

    // The Moon
    MOON_MINE, // 29
    MOON_HABITAT, // 30
    MOON_ROAD, // 31
    LUNA_TRADE_STATION, // 32
    LUNA_MINING_HUB, // 33
    LUNA_TRAIN_STATION, // 34
    LUNAR_MINE_URBANIZATION, // 35

    // Pathfinders
    WETLANDS, // 36
    RED_CITY, // 37
    MARTIAN_NATURE_WONDERS, // 38
    CRASHLANDING, // 39

    MARS_NOMADS, // 40
    REY_SKYWALKER, // 41

    // Underworld
    MAN_MADE_VOLCANO, // 42

    // Promo
    NEW_HOLLAND, // 43
  }

  export type UndergroundResourceToken =
  'nothing' |
  'card1' | 'card2' |
  'corruption1' | 'corruption2' |
  'data1' | 'data2' | 'data3' |
  'steel2' | 'steel1production' |
  'titanium2' | 'titanium1production' |
  'plant1' | 'plant2' | 'plant3' | 'plant1production' |
  'titaniumandplant' |
  'energy1production' | 'heat2production' |
  'microbe1' | 'microbe2' | 'tr' | 'ocean' |
  'data1pertemp' | 'microbe1pertemp' | 'plant2pertemp' | 'steel2pertemp' | 'titanium1pertemp';

export type SpaceModel = {
  id: SpaceId;
  x: number;
  y: number;
  spaceType: SpaceType;

  bonus: Array<SpaceBonus>;
  color?: Color;
  tileType?: TileType;
  highlight?: SpaceHighlight;
  rotated?: true; // Absent or true
  gagarin?: number; // 0 means current
  cathedral?: true; // Absent or true
  nomads?: true; // Absent or true
  coOwner?: Color;

  undergroundResources?: UndergroundResourceToken;
  excavator?: Color;
}

export type MoonModel = {
  spaces: Array<SpaceModel>;
  habitatRate: number;
  miningRate: number;
  logisticsRate: number;
}

export type PathfindersModel = {
  venus: number;
  earth: number;
  mars: number;
  jovian: number;
  moon: number;
}

export enum Phase {
  /**
   * Not part of the rulebook, initial drafting includes project cards and
   * prelude cards (maybe others ongoing?) Transitions to RESEARCH
   * but as mentioned above, only the first generation type of research.
   */
  INITIALDRAFTING = 'initial_drafting',

  /** Between 1st gen research and action phases, each player plays their preludes. */
  PRELUDES = 'preludes',
  /** Between 1st gen research and action phases, each player plays their CEOs. */
  CEOS = 'ceos',

  /**
   * The phase where a player chooses cards to keep.
   * This includes the first generation drafting phase, which has different
   * behavior and transitions to a different eventual phase
   */
  RESEARCH = 'research',

  /** The standard drafting phase, as described by the official rules variant. */
  DRAFTING = 'drafting',

  /** Maps to rulebook action phase */
  ACTION = 'action',

  /** Maps to rulebook production phase */
  PRODUCTION = 'production',
  /** Standard rulebook Solar phase, triggers WGT, and final greeneries, but not Turmoil. */
  SOLAR = 'solar',
  /** Does some cleanup and also executes the rulebook's turn order phase. */
  INTERGENERATION = 'intergeneration',

  /** The game is over. */
  END = 'end',
}

export enum Tag {
    BUILDING = 'building',
    SPACE = 'space',
    SCIENCE = 'science',
    POWER = 'power',
    EARTH = 'earth',
    JOVIAN = 'jovian',
    VENUS = 'venus',
    PLANT = 'plant',
    MICROBE = 'microbe',
    ANIMAL = 'animal',
    CITY = 'city',
    MOON = 'moon',
    MARS = 'mars',
    CRIME = 'crime',
    WILD = 'wild',
    EVENT = 'event',
    CLONE = 'clone',
}

export enum PartyName {
    MARS = 'Mars First',
    SCIENTISTS = 'Scientists',
    UNITY = 'Unity',
    KELVINISTS = 'Kelvinists',
    REDS = 'Reds',
    GREENS = 'Greens'
}

export type DelegatesModel = {
  color: Color;
  number: number;
}

export type PartyModel = {
  name: PartyName;
  partyLeader: Color | undefined;
  delegates: Array<DelegatesModel>;
}

export enum GlobalEventName {
    GLOBAL_DUST_STORM = 'Global Dust Storm',
    SPONSORED_PROJECTS = 'Sponsored Projects',
    ASTEROID_MINING = 'Asteroid Mining',
    GENEROUS_FUNDING = 'Generous Funding',
    SUCCESSFUL_ORGANISMS = 'Successful Organisms',
    ECO_SABOTAGE = 'Eco Sabotage',
    PRODUCTIVITY = 'Productivity',
    SNOW_COVER = 'Snow Cover',
    DIVERSITY = 'Diversity',
    PANDEMIC = 'Pandemic',
    WAR_ON_EARTH = 'War on Earth',
    IMPROVED_ENERGY_TEMPLATES = 'Improved Energy Templates',
    INTERPLANETARY_TRADE = 'Interplanetary Trade',
    CELEBRITY_LEADERS = 'Celebrity Leaders',
    SPINOFF_PRODUCTS = 'Spin-Off Products',
    ELECTION = 'Election',
    AQUIFER_RELEASED_BY_PUBLIC_COUNCIL = 'Aquifer Released by Public Council',
    PARADIGM_BREAKDOWN = 'Paradigm Breakdown',
    HOMEWORLD_SUPPORT = 'Homeworld Support',
    RIOTS = 'Riots',
    VOLCANIC_ERUPTIONS = 'Volcanic Eruptions',
    MUD_SLIDES = 'Mud Slides',
    MINERS_ON_STRIKE = 'Miners On Strike',
    SABOTAGE = 'Sabotage',
    REVOLUTION = 'Revolution',
    DRY_DESERTS = 'Dry Deserts',
    SCIENTIFIC_COMMUNITY = 'Scientific Community',
    CORROSIVE_RAIN = 'Corrosive Rain',
    JOVIAN_TAX_RIGHTS = 'Jovian Tax Rights',
    RED_INFLUENCE = 'Red Influence',
    SOLARNET_SHUTDOWN = 'Solarnet Shutdown',
    STRONG_SOCIETY = 'Strong Society',
    SOLAR_FLARE = 'Solar Flare',
    VENUS_INFRASTRUCTURE = 'Venus Infrastructure',
    CLOUD_SOCIETIES = 'Cloud Societies',
    MICROGRAVITY_HEALTH_PROBLEMS = 'Microgravity Health Problems',

    // Community
    LEADERSHIP_SUMMIT = 'Leadership Summit',

    // Pathfinders
    BALANCED_DEVELOPMENT = 'Balanced Development',
    CONSTANT_STRUGGLE = 'Constant Struggle',
    TIRED_EARTH = 'Tired Earth',
    MAGNETIC_FIELD_STIMULATION_DELAYS ='Magnetic Field Stimulation Delays',
    COMMUNICATION_BOOM = 'Communication Boom',
    SPACE_RACE_TO_MARS = 'Space Race to Mars',

    // Underworld
    LAGGING_REGULATION = 'Lagging Regulation',
    FAIR_TRADE_COMPLAINT = 'Fair Trade Complaint',
    MIGRATION_UNDERGROUND = 'Migration Underground',
    SEISMIC_PREDICTIONS = 'Seismic Predictions',
    MEDIA_STIR = 'Media Stir',
}

export type PoliticalAgendasModel = {
  marsFirst: Agenda;
  scientists: Agenda;
  unity: Agenda;
  greens: Agenda;
  reds: Agenda;
  kelvinists: Agenda;
}

export type PolicyUser = {
  color: Color;
  turmoilPolicyActionUsed: boolean;
  politicalAgendasActionUsedCount: number;
}

export type TurmoilModel = {
  dominant: PartyName | undefined;
  ruling: PartyName | undefined;
  chairman: Color | undefined;
  parties: Array<PartyModel>;
  lobby: Array<Color>;
  reserve: Array<DelegatesModel>;
  distant: GlobalEventName | undefined;
  coming: GlobalEventName | undefined;
  current: GlobalEventName | undefined;
  politicalAgendas: PoliticalAgendasModel | undefined;
  policyActionUsers: Array<PolicyUser>;
}

export type GameModel = {
  aresData: AresData | undefined;
  awards: ReadonlyArray<FundedAwardModel>;
  colonies: ReadonlyArray<ColonyModel>;
  discardedColonies: ReadonlyArray<ColonyName>;
  deckSize: number;
  expectedPurgeTimeMs: number;
  experimentalReset?: boolean;
  gameAge: number;
  gameOptions: GameOptionsModel;
  generation: number;
  globalsPerGeneration: ReadonlyArray<Partial<Record<GlobalParameter, number>>>,
  isSoloModeWin: boolean;
  lastSoloGeneration: number,
  milestones: ReadonlyArray<ClaimedMilestoneModel>;
  moon: MoonModel | undefined;
  oceans: number;
  oxygenLevel: number;
  passedPlayers: ReadonlyArray<Color>;
  pathfinders: PathfindersModel | undefined;
  phase: Phase;
  spaces: ReadonlyArray<SpaceModel>;
  spectatorId?: SpectatorId;
  step: number;
  tags: ReadonlyArray<Tag>;
  temperature: number;
  isTerraformed: boolean;
  turmoil: TurmoilModel | undefined;
  undoCount: number;
  venusScaleLevel: number;
}

export type Protection = 'off' | 'on' | 'half';

export enum Resource {
    MEGACREDITS = 'megacredits',
    STEEL = 'steel',
    TITANIUM = 'titanium',
    PLANTS = 'plants',
    ENERGY = 'energy',
    HEAT = 'heat'
}

export type TimerModel = {
  sumElapsed: number;
  startedAt: number;
  running: boolean;
  afterFirstAction: boolean;
  lastStoppedAt: number;
}

export type MADetail = {message: string, messageArgs?: Array<string>, victoryPoint: number};

export type VictoryPointsBreakdown = {
  terraformRating: number;
  milestones: number;
  awards: number;
  greenery: number;
  city: number;
  escapeVelocity: number;
  moonHabitats: number;
  moonMines: number;
  moonRoads: number;
  planetaryTracks: number;
  victoryPoints: number;
  total: number;
  detailsCards: ReadonlyArray<{cardName: string, victoryPoint: number}>;
  detailsMilestones: ReadonlyArray<MADetail>;
  detailsAwards: ReadonlyArray<MADetail>;
  detailsPlanetaryTracks: ReadonlyArray<{tag: Tag, points: number}>;
  // Total VP less than 0. For Underworld
  negativeVP: number;
}

type AlliedPartyModel = {
  partyName: PartyName;
  agenda: Agenda;
};

export type PublicPlayerModel = {
  actionsTakenThisRound: number;
  actionsThisGeneration: ReadonlyArray<CardName>;
  actionsTakenThisGame: number;
  availableBlueCardActionCount: number;
  cardCost: number;
  cardDiscount: number;
  cardsInHandNbr: number;
  citiesCount: number;
  coloniesCount: number;
  color: Color;
  corruption: number,
  energy: number;
  energyProduction: number;
  excavations: number,
  fleetSize: number;
  handicap: number | undefined;
  heat: number;
  heatProduction: number;
  id: PlayerId | undefined;
  influence: number;
  isActive: boolean;
  lastCardPlayed?: CardName;
  megaCredits: number;
  megaCreditProduction: number;
  name: string;
  needsToDraft: boolean | undefined;
  needsToResearch: boolean | undefined;
  noTagsCount: number;
  plants: number;
  plantProduction: number;
  protectedResources: Record<Resource, Protection>;
  protectedProduction: Record<Resource, Protection>;
  tableau: ReadonlyArray<CardModel>;
  selfReplicatingRobotsCards: Array<CardModel>;
  steel: number;
  steelProduction: number;
  steelValue: number;
  tags: Record<Tag, number>
  terraformRating: number;
  timer: TimerModel;
  titanium: number;
  titaniumProduction: number;
  titaniumValue: number;
  tradesThisGeneration: number;
  victoryPointsBreakdown: VictoryPointsBreakdown;
  victoryPointsByGeneration: ReadonlyArray<number>;
  alliedParty?: AlliedPartyModel;
}

export interface ViewModel {
  game: GameModel;
  players: Array<PublicPlayerModel>;
  id?: ParticipantId;
  thisPlayer: PublicPlayerModel | undefined;
  runId: string;
}

export type CardDiscount = {
  /** The tag this discount applies to, or when undefined, it applies to all cards. */
  tag?: Tag,
  /** The M€ discount. */
  amount: number,
  /** Describes whether the discount is applied once for the card, or for ever tag. */
  per?: 'card' | 'tag',
 }


export type LogMessageDataAttrs = {
  /** When true for a card, also show the card's tags */
  tags?: boolean,
  /** When true for a card, also show the card's cost */
  cost?: boolean,
}
export enum LogMessageDataType {
  STRING, // 0
  RAW_STRING, // Raw strings are untranslated.  // 1
  PLAYER, // 2
  CARD, // 3
  AWARD, // 4
  MILESTONE, // 5
  COLONY, // 6
  _STANDARD_PROJECT, // 7 // NO LONGER USED
  PARTY, // 8
  TILE_TYPE, // 9
  SPACE_BONUS, // 10
  GLOBAL_EVENT, // 11
}

type Types = {
  type: LogMessageDataType.STRING | LogMessageDataType.RAW_STRING,
  value: string,
} | {
  type: LogMessageDataType.PLAYER,
  value: Color,
} | {
  type: LogMessageDataType.CARD,
  value: CardName,
} | {
  type: LogMessageDataType.AWARD,
  value: AwardName,
} | {
  type: LogMessageDataType.MILESTONE,
  value: MilestoneName,
} | {
  type: LogMessageDataType.COLONY,
  value: ColonyName,
} | {
  type: LogMessageDataType.PARTY,
  value: PartyName,
} | {
  type: LogMessageDataType.TILE_TYPE,
  value: TileType,
} | {
  type: LogMessageDataType.SPACE_BONUS,
  value: SpaceBonus,
} | {
  type: LogMessageDataType.PARTY,
  value: PartyName,
} | {
  type:
  LogMessageDataType.GLOBAL_EVENT;
  value: GlobalEventName,
};

export type LogMessageData = Types & {
  attrs?: LogMessageDataAttrs;
}

export interface Message {
  data: Array<LogMessageData>;
  message: string;
}

export type Warning =
 'maxtemp' |
 'maxoxygen' |
 'maxoceans' |
 'maxvenus' |
 'maxHabitatRate' |
 'maxMiningRate' |
 'maxLogisticsRate' |
 'decreaseOwnProduction' |
 'removeOwnPlants' |
 'buildOnLuna' |
 'preludeFizzle' |
 'underworldMustExcavateEnergy' |
 'deckTooSmall' |
 'cannotAffordBoardOfDirectors' |
 'marsIsTerraformed' |
 'ineffectiveDoubleDown' |
 'noMatchingCards' |
 'unusableEventsForAstraMechanica' |
 'noEffect' |
 'selfTarget';

 export type Units = {
  megacredits: number;
  steel: number;
  titanium: number;
  plants: number;
  energy: number;
  heat: number;
}

export interface CardModel {
    name: CardName;
    resources: number | undefined;
    calculatedCost?: number;
    isSelfReplicatingRobotsCard?: boolean,
    discount?: Array<CardDiscount>,
    isDisabled?: boolean; // Used with Pharmacy Union
    warning?: string | Message;
    warnings?: ReadonlyArray<Warning>;
    reserveUnits?: Readonly<Units>; // Written for The Moon, but useful in other contexts.
    bonusResource?: Array<Resource>; // Used with the Mining cards and Robotic Workforce
    cloneTag?: Tag; // Used with Pathfinders
}

export type BaseInputModel = {
  title: string | Message;
  warning?: string | Message;
  buttonLabel: string;
}

export type AndOptionsModel = BaseInputModel & {
  type: 'and';
  options: Array<PlayerInputModel>;
}

export type OrOptionsModel = BaseInputModel & {
  type: 'or';
  options: Array<PlayerInputModel>;
  // When set, initialIdx represents the option within `options` that should be
  // shows as the default selection.
  initialIdx?: number;
}

export type SelectInitialCardsModel = BaseInputModel & {
  type: 'initialCards';
  options: Array<PlayerInputModel>;
}

export type SelectOptionModel = BaseInputModel & {
  type: 'option';
  warnings?: Array<Warning>;
}

export type SelectProjectCardToPlayModel = BaseInputModel & {
  type: 'projectCard';
  cards: ReadonlyArray<CardModel>;
  paymentOptions: Partial<PaymentOptions>,
  microbes: number;
  floaters: number;
  lunaArchivesScience: number;
  seeds: number;
  graphene: number;
  kuiperAsteroids: number;
  corruption: number;
}

export type SelectCardModel = BaseInputModel & {
  type: 'card';
  cards: ReadonlyArray<CardModel>;
  max: number;
  min: number;
  showOnlyInLearnerMode: boolean;
  selectBlueCardAction: boolean;
  showOwner: boolean;
}

export type SelectColonyModel = BaseInputModel & {
  type: 'colony';
  coloniesModel: ReadonlyArray<ColonyModel>;
}

export type SelectPaymentModel = BaseInputModel & {
  type: 'payment';
  amount: number;
  paymentOptions: Partial<PaymentOptions>;
  seeds: number;
  auroraiData: number;
  kuiperAsteroids: number;
  spireScience: number;
}

export type SelectPlayerModel = BaseInputModel & {
  type: 'player';
  players: ReadonlyArray<Color>;
}

export type SelectSpaceModel = BaseInputModel & {
  type: 'space';
  spaces: ReadonlyArray<SpaceId>;
}

export type SelectAmountModel = BaseInputModel & {
  type: 'amount';
  min: number;
  max: number;
  maxByDefault: boolean;
}

export type SelectDelegateModel = BaseInputModel & {
  type: 'delegate';
  players: Array<ColorWithNeutral>;
}

export type SelectPartyModel = BaseInputModel & {
  type: 'party';
  parties: Array<PartyName>;
}
export type PayProductionModel = {
    cost: number;
    units: Units;
}

export type SelectProductionToLoseModel = BaseInputModel & {
  type: 'productionToLose';
  payProduction: PayProductionModel;
}

export type ShiftAresGlobalParametersModel = BaseInputModel & {
  type: 'aresGlobalParameters';
  aresData: AresData;
}

export type SelectGlobalEventModel = BaseInputModel & {
  type: 'globalEvent';
  globalEventNames: Array<GlobalEventName>;
}

export type SelectResourceModel = BaseInputModel & {
  type: 'resource';
  include: ReadonlyArray<keyof Units>;
}

export type SelectResourcesModel = BaseInputModel & {
  type: 'resources';
  count: number;
}

export type PlayerInputModel =
  AndOptionsModel |
  OrOptionsModel |
  SelectInitialCardsModel |
  SelectOptionModel |
  SelectProjectCardToPlayModel |
  SelectCardModel |
  SelectAmountModel |
  SelectCardModel |
  SelectColonyModel |
  SelectDelegateModel |
  SelectPartyModel |
  SelectPaymentModel |
  SelectPlayerModel |
  SelectProductionToLoseModel |
  SelectProjectCardToPlayModel |
  SelectSpaceModel |
  ShiftAresGlobalParametersModel |
  SelectGlobalEventModel |
  SelectResourceModel |
  SelectResourcesModel;

export interface PlayerViewModel extends ViewModel {
  autopass: boolean;
  cardsInHand: ReadonlyArray<CardModel>;
  dealtCorporationCards: ReadonlyArray<CardModel>;
  dealtPreludeCards: ReadonlyArray<CardModel>;
  dealtProjectCards: ReadonlyArray<CardModel>;
  dealtCeoCards: ReadonlyArray<CardModel>;
  draftedCards: ReadonlyArray<CardModel>;
  id: PlayerId;
  ceoCardsInHand: ReadonlyArray<CardModel>;
  pickedCorporationCard: ReadonlyArray<CardModel>; // Why Array?
  preludeCardsInHand: ReadonlyArray<CardModel>;
  thisPlayer: PublicPlayerModel;
  waitingFor: PlayerInputModel | undefined;
}
