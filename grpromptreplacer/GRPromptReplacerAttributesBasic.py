import re
import random


class GRPromptReplacerAttributesBasic:
    """
    A ComfyUI node that replaces specific facial/appearance attributes (hair colour,
    skin tone, hair style, eye colour, smile, facial expression, pose, hand position,
    head position, and clothing) in a prompt using dropdown selectors.

    Each category has a single dropdown of target values. The phrase-detection
    mappings for each category (what source phrases get swapped in) run entirely
    in the background using built-in tables — nothing to configure.

    Set a category's dropdown to "none" to leave that attribute untouched.
    Use the "randomize" checkbox and "seed" value to generate random combinations.

    Special Categories:
    - Photograph Type: Intelligently inserts distance/shot type with fallback (e.g., "close up photograph")
    """

    # ---- HEADWEAR ------------------------------------------------------------
    HEADWEAR_OPTIONS = [
        "none",
        "hijab", "headscarf", "head covering", "head wrap",
        "beanie", "cap", "baseball cap", "hat", "wide-brimmed hat",
        "cowboy hat", "sailor hat", "beret", "fedora", "headband",
        "bandana", "headpiece", "turban", "veil",
        # Adding more detailed options based on prompt analysis
        "black hijab", "blue hijab", "green hijab", "brown hijab", "white hijab",
        "pink hijab", "red hijab", "maroon hijab",
        "beaded headpiece", "gold headpiece", "ornate headpiece",
        "knitted beanie", "fur hat", "pink beanie", "red beanie",
    ]
    HEADWEAR_MAPPING_DEFAULT = (
        # --- Head Coverings (Hijab/Headscarves) ---
        "wearing a hijab, wearing a {value}\n"
        "wearing a black hijab, wearing a {value}\n"
        "wearing a blue hijab, wearing a {value}\n"
        "wearing a brown hijab, wearing a {value}\n"
        "wearing a green hijab, wearing a {value}\n"
        "wearing a pink hijab, wearing a {value}\n"
        "wearing a red hijab, wearing a {value}\n"
        "wearing a white hijab, wearing a {value}\n"
        "wearing a maroon hijab, wearing a {value}\n"
        "black hijab,{value}\n"
        "blue hijab,{value}\n"
        "brown hijab,{value}\n"
        "green hijab,{value}\n"
        "pink hijab,{value}\n"
        "red hijab,{value}\n"
        "white hijab,{value}\n"
        "maroon hijab,{value}\n"
        "a hijab, a {value}\n"
        "the hijab, the {value}\n"
        "with a hijab, with a {value}\n"
        # --- Hats & Headpieces ---
        "wearing a hat, wearing a {value}\n"
        "wearing a beanie, wearing a {value}\n"
        "wearing a cap, wearing a {value}\n"
        "wearing a baseball cap, wearing a {value}\n"
        "wearing a cowboy hat, wearing a {value}\n"
        "wearing a wide-brimmed hat, wearing a {value}\n"
        "wearing a beret, wearing a {value}\n"
        "wearing a bandana, wearing a {value}\n"
        "wearing a headband, wearing a {value}\n"
        "wearing a headpiece, wearing a {value}\n"
        "wearing a gold headpiece, wearing a {value}\n"
        "wearing a beaded headpiece, wearing a {value}\n"
        "wearing an ornate headpiece, wearing a {value}\n"
        "a hat, a {value}\n"
        "the hat, the {value}\n"
        "with a hat, with a {value}\n"
        "beanie,{value}\n"
        "cap,{value}\n"
        "baseball cap,{value}\n"
        "cowboy hat,{value}\n"
        "wide-brimmed hat,{value}\n"
        "beret,{value}\n"
        "fedora,{value}\n"
        "headband,{value}\n"
        "bandana,{value}\n"
        "turban,{value}\n"
        "veil,{value}\n"
        # --- Contextual Replacements ---
        "headscarf,{value}\n"
        "head wrap,{value}\n"
        "head covering,{value}\n"
        "headpiece,{value}\n"
        "hair accessory,{value}\n"
        # --- Specific Options from Prompts ---
        "red beanie,{value}\n"
        "pink beanie,{value}\n"
        "knitted beanie,{value}\n"
        "fur hat,{value}\n"
        "sailor hat,{value}\n"
        "black headscarf,{value}\n"
        "blue headscarf,{value}\n"
        "brown headscarf,{value}\n"
        "green headscarf,{value}\n"
        "pink headscarf,{value}\n"
        "red headscarf,{value}\n"
        "white headscarf,{value}\n"
        "maroon headscarf,{value}\n"
        "gold headpiece,{value}\n"
        "beaded headpiece,{value}\n"
        "ornate headpiece,{value}"
    )

    # ---- PHOTOGRAPH TYPE (with intelligent fallback) ------------------------
    
    PHOTOGRAPH_TYPE_OPTIONS = [
        "none",
        # Distance/Shot types
        "close up", "extreme close up", "macro", "micro",
        "medium shot", "medium close up", "medium full shot",
        "full shot", "full body", "whole body",
        "long shot", "wide shot", "extreme wide shot",
        "distant", "distance", "from a distance", "far away",
        "panoramic", "panorama",
        # Angle/Camera position
        "low angle", "high angle", "eye level", "birds eye",
        "worm's eye", "overhead", "aerial", "drone",
        # Composition
        "portrait", "headshot", "bust", "half body", "three quarter",
        "candid", "action shot", "motion", "dynamic",
        # Special
        "selfie", "group photo", "family portrait", "wedding photo",
        "lifestyle", "editorial", "fashion", "glamour",
        "boudoir", "intimate", "artistic", "fine art",
        "documentary", "street photography", "travel", "landscape",
        "still life", "product shot", "food photography",
        "macro photography", "wildlife", "nature", "architectural",
        "interior", "exterior", "night photography", "long exposure",
        "silhouette", "reflection", "shadow", "backlit",
        # Environment
        "indoor", "outdoor", "studio", "location", "on location",
        "environmental", "in situ", "natural light", "studio light",
        "window light", "golden hour", "blue hour", "sunset", "sunrise",
        "overcast", "cloudy", "clear sky", "stormy", "foggy", "misty"
    ]
    
    # Mapping for finding and replacing photograph type descriptors
    PHOTOGRAPH_TYPE_MAPPING_DEFAULT = (
        # Common photograph descriptors - these will be swapped
        "close up photograph,{value} photograph\n"
        "close-up photograph,{value} photograph\n"
        "extreme close up,{value}\n"
        "extreme close-up,{value}\n"
        "macro photograph,{value} photograph\n"
        "distant photograph,{value} photograph\n"
        "distance photograph,{value} photograph\n"
        "wide shot,{value} shot\n"
        "full shot,{value} shot\n"
        "full body shot,{value} shot\n"
        "long shot,{value} shot\n"
        "medium shot,{value} shot\n"
        "portrait photograph,{value} photograph\n"
        "headshot,{value}\n"
        "head shot,{value}\n"
        "bust shot,{value} shot\n"
        "candid photograph,{value} photograph\n"
        "action shot,{value} shot\n"
        "selfie,{value} selfie\n"
        "close up,{value}\n"
        "close-up,{value}\n"
        "distant,{value}\n"
        # Fallback patterns - these will be replaced with the full phrase
        "a photograph of,a {value} photograph of\n"
        "a photo of,a {value} photo of\n"
        "an image of,a {value} image of\n"
        "a picture of,a {value} picture of\n"
        "photograph of,{value} photograph of\n"
        "photo of,{value} photo of\n"
        "image of,{value} image of\n"
        "picture of,{value} picture of\n"
        "studio photograph,{value} photograph\n"
        "outdoor photograph,{value} photograph\n"
        "indoor photograph,{value} photograph\n"
        "natural light photograph,{value} photograph\n"
        "golden hour photograph,{value} photograph\n"
        "black and white photograph,{value} photograph\n"
        "color photograph,{value} photograph\n"
        "vintage photograph,{value} photograph\n"
        "polaroid photograph,{value} photograph\n"
        "film photograph,{value} photograph\n"
        "digital photograph,{value} photograph"
    )

    # ---- HAIR COLOR ---------------------------------------------------------
    HAIR_COLOR_OPTIONS = [
        "none", "blonde", "brunette", "black", "dark brown", "light brown",
        "red", "auburn", "gray", "white", "green", "blue", "pink", "purple",
        "turquoise", "silver", "caramel",
    ]
    HAIR_COLOR_MAPPING_DEFAULT = (
        "dark brown hair,{value} hair\n"
        "blonde hair,{value} hair\n"
        "blond hair,{value} hair\n"
        "light brown hair,{value} hair\n"
        "reddish-brown hair,{value} hair\n"
        "reddish-orange hair,{value} hair\n"
        "dark hair,{value} hair\n"
        "brunette hair,{value} hair\n"
        "black hair,{value} hair\n"
        "brown hair,{value} hair\n"
        "red hair,{value} hair\n"
        "auburn hair,{value} hair\n"
        "gray hair,{value} hair\n"
        "grey hair,{value} hair\n"
        "gray-streaked hair,{value} hair\n"
        "blue-streaked hair,{value} hair\n"
        "white hair,{value} hair\n"
        "green hair,{value} hair\n"
        "blue hair,{value} hair\n"
        "pink hair,{value} hair\n"
        "purple hair,{value} hair\n"
        "turquoise hair,{value} hair\n"
        "silver hair,{value} hair\n"
        "caramel-colored hair,{value} hair\n"
        "light blonde,{value}\n"
        "dark blonde,{value}\n"
        "light brunette,{value}\n"
        "dark brunette,{value}\n"
        "blonde,{value}\n"
        "brunette,{value}\n"
        "redhead,{value}\n"
        "ginger,{value}"
    )

    # ---- SKIN TONE ----------------------------------------------------------
    SKIN_TONE_OPTIONS = [
        "none", "pale", "fair", "light", "medium", "tan", "olive", "dark",
        "deep", "freckled",
    ]
    SKIN_TONE_MAPPING_DEFAULT = (
        "light skin,{value} skin\n"
        "pale skin,{value} skin\n"
        "fair skin,{value} skin\n"
        "dark skin,{value} skin\n"
        "tan skin,{value} skin\n"
        "olive skin,{value} skin\n"
        "medium skin,{value} skin\n"
        "freckled skin,{value} skin\n"
        "medium-brown skin,{value} skin\n"
        "medium-tan skin,{value} skin\n"
        "medium-dark skin,{value} skin\n"
        "fair complexion,{value} complexion\n"
        "dark complexion,{value} complexion\n"
        "medium complexion,{value} complexion\n"
        "skin tone is light,skin tone is {value}\n"
        "skin tone is pale,skin tone is {value}\n"
        "skin tone is fair,skin tone is {value}\n"
        "skin tone is dark,skin tone is {value}\n"
        "skin tone is tan,skin tone is {value}\n"
        "skin tone is medium,skin tone is {value}\n"
        "skin tone is medium-brown,skin tone is {value}\n"
        "skin tone is medium-tan,skin tone is {value}\n"
        "skin tone is medium-dark,skin tone is {value}"
    )

    # ---- HAIR STYLE ---------------------------------------------------------
    HAIR_STYLE_OPTIONS = [
        "none", "long straight", "long wavy", "long curly", "shoulder-length",
        "short", "bob", "pixie", "ponytail", "high ponytail", "low ponytail",
        "braided", "braids", "bun", "messy bun", "updo", "sleek updo",
        "bangs", "fringe", "afro", "pigtails", "buzz cut",
    ]
    HAIR_STYLE_MAPPING_DEFAULT = (
        "shoulder-length,{value}\n"
        "long straight,{value}\n"
        "long wavy,{value}\n"
        "long curly,{value}\n"
        "loose waves,{value}\n"
        "loose curls,{value}\n"
        "short hair,{value} hair\n"
        "bob haircut,{value} haircut\n"
        "bob cut,{value} cut\n"
        "pixie cut,{value}\n"
        "ponytail,{value}\n"
        "high ponytail,{value}\n"
        "low ponytail,{value}\n"
        "long ponytail,{value}\n"
        "braided hair,{value} hair\n"
        "braids,{value}\n"
        "two braids,{value}\n"
        "bun,{value}\n"
        "messy bun,{value}\n"
        "neat bun,{value}\n"
        "updo,{value}\n"
        "sleek updo,{value}\n"
        "elegant updo,{value}\n"
        "messy updo,{value}\n"
        "bangs,{value}\n"
        "fringe,{value}\n"
        "afro,{value}\n"
        "pigtails,{value}\n"
        "ponytail style haircut,{value}\n"
        "ponytail style,{value}\n"
        "ponytail hairstyle,{value}\n"
        "bun style haircut,{value}\n"
        "updo style haircut,{value}\n"
        "braided style haircut,{value}"
    )

    # ---- EYE COLOR ----------------------------------------------------------
    EYE_COLOR_OPTIONS = [
        "none", "blue", "green", "brown", "hazel", "gray", "amber", "violet",
    ]
    EYE_COLOR_MAPPING_DEFAULT = (
        "blue eyes,{value} eyes\n"
        "green eyes,{value} eyes\n"
        "brown eyes,{value} eyes\n"
        "hazel eyes,{value} eyes\n"
        "gray eyes,{value} eyes\n"
        "grey eyes,{value} eyes\n"
        "dark eyes,{value} eyes\n"
        "light-colored eyes,{value} eyes\n"
        "eyes are blue,eyes are {value}\n"
        "eyes are green,eyes are {value}\n"
        "eyes are brown,eyes are {value}\n"
        "eyes are hazel,eyes are {value}\n"
        "eyes are gray,eyes are {value}\n"
        "eyes are grey,eyes are {value}\n"
        "eyes are dark,eyes are {value}"
    )

    # ---- SMILE --------------------------------------------------------------
    SMILE_OPTIONS = [
        "none", "big smile", "subtle smile", "slight smile", "warm smile",
        "gentle smile", "confident smile", "cheerful smile", "playful smile",
        "friendly smile", "radiant smile", "mischievous smile", "broad smile",
        "no smile", "smirk", "closed-mouth smile", "toothy grin",
    ]
    SMILE_MAPPING_DEFAULT = (
        "big smile,{value}\n"
        "subtle smile,{value}\n"
        "slight smile,{value}\n"
        "warm smile,{value}\n"
        "gentle smile,{value}\n"
        "confident smile,{value}\n"
        "cheerful smile,{value}\n"
        "playful smile,{value}\n"
        "friendly smile,{value}\n"
        "bright smile,{value}\n"
        "radiant smile,{value}\n"
        "inviting smile,{value}\n"
        "mischievous smile,{value}\n"
        "broad smile,{value}\n"
        "wide smile,{value}\n"
        "crooked smile,{value}\n"
        "closed-mouth smile,{value}\n"
        "no smile,{value}\n"
        "smirk,{value}\n"
        "toothy grin,{value}"
    )

    # ---- EXPRESSION ---------------------------------------------------------
    EXPRESSION_OPTIONS = [
        "none", "neutral", "serious", "relaxed", "playful", "confident",
        "cheerful", "peaceful", "serene", "focused", "surprised", "curious",
        "content", "calm", "thoughtful", "contemplative", "happy", "friendly",
        "joyful", "amused", "pensive", "sultry", "seductive", "alluring",
        "sensual",
    ]
    EXPRESSION_MAPPING_DEFAULT = (
        "neutral expression,{value} expression\n"
        "serious expression,{value} expression\n"
        "relaxed expression,{value} expression\n"
        "playful expression,{value} expression\n"
        "confident expression,{value} expression\n"
        "cheerful expression,{value} expression\n"
        "peaceful expression,{value} expression\n"
        "serene expression,{value} expression\n"
        "focused expression,{value} expression\n"
        "surprised expression,{value} expression\n"
        "curious expression,{value} expression\n"
        "content expression,{value} expression\n"
        "calm expression,{value} expression\n"
        "thoughtful expression,{value} expression\n"
        "contemplative expression,{value} expression\n"
        "happy expression,{value} expression\n"
        "friendly expression,{value} expression\n"
        "joyful expression,{value} expression\n"
        "mischievous expression,{value} expression\n"
        "amused expression,{value} expression\n"
        "pensive expression,{value} expression\n"
        "shocked expression,{value} expression\n"
        "sultry expression,{value} expression\n"
        "seductive expression,{value} expression\n"
        "alluring expression,{value} expression\n"
        "sensual expression,{value} expression\n"
        "facial expression,{value} expression"
    )

    # ---- LIGHTING -----------------------------------------------------------
    LIGHTING_OPTIONS = [
        "none", "soft", "bright", "natural", "warm", "dim", "even", "diffused",
        "dramatic", "artificial", "ambient", "harsh",
        "low-key", "moody", "eerie", "ominous", "sinister", "creepy",
        "foreboding", "shadowy", "grim", "spooky", "haunting", "candlelit",
        "flickering", "moonlit", "noir", "chiaroscuro",
        "gothic", "cinematic", "filmic", "volumetric", "atmospheric",
        "dreamlike", "surreal", "ethereal", "magical", "mystical", "arcane",
        "celestial", "divine", "cyberpunk", "dystopian", "post apocalyptic",
    ]
    LIGHTING_MAPPING_DEFAULT = (
        "lighting is soft and natural,lighting is {value}\n"
        "lighting is bright and natural,lighting is {value}\n"
        "lighting is soft and warm,lighting is {value}\n"
        "lighting is bright and even,lighting is {value}\n"
        "lighting is soft and even,lighting is {value}\n"
        "lighting is even and bright,lighting is {value}\n"
        "lighting is soft and diffused,lighting is {value}\n"
        "lighting is even and soft,lighting is {value}\n"
        "lighting is bright and sunny,lighting is {value}\n"
        "lighting is natural and bright,lighting is {value}\n"
        "lighting is soft and evenly distributed,lighting is {value}\n"
        "lighting is even and diffuse,lighting is {value}\n"
        "lighting is warm and soft,lighting is {value}\n"
        "lighting is bright and evenly distributed,lighting is {value}\n"
        "lighting is dramatic,lighting is {value}\n"
        "lighting is dim,lighting is {value}\n"
        "lighting is bright,lighting is {value}\n"
        "lighting is natural,lighting is {value}\n"
        "lighting is even,lighting is {value}\n"
        "lighting is artificial,lighting is {value}\n"
        "soft lighting,{value} lighting\n"
        "warm lighting,{value} lighting\n"
        "natural lighting,{value} lighting\n"
        "dramatic lighting,{value} lighting\n"
        "volumetric lighting,{value} lighting\n"
        "ambient lighting,{value} lighting\n"
        "artificial lighting,{value} lighting\n"
        "indoor lighting,{value} lighting\n"
        "warm ambient lighting,{value} lighting\n"
        "soft ambient lighting,{value} lighting\n"
        "subtle rim lighting,{value} lighting\n"
        "high contrast lighting,{value} lighting\n"
        "dramatic cinematic lighting,{value} lighting\n"
        "natural daylight,{value} light\n"
        "warm tungsten light,{value} light\n"
        "cool blue light,{value} light\n"
        "eerie,{value}\n"
        "ominous,{value}\n"
        "haunting,{value}\n"
        "sinister,{value}\n"
        "gothic,{value}\n"
        "cinematic,{value}\n"
        "filmic,{value}\n"
        "volumetric,{value}\n"
        "atmospheric,{value}\n"
        "moody,{value}\n"
        "dreamlike,{value}\n"
        "surreal,{value}\n"
        "ethereal,{value}\n"
        "magical,{value}\n"
        "mystical,{value}\n"
        "arcane,{value}\n"
        "celestial,{value}\n"
        "divine,{value}\n"
        "noir,{value}\n"
        "cyberpunk,{value}\n"
        "dystopian,{value}\n"
        "post apocalyptic,{value}\n"
        "softly lit,{value}\n"
        "brightly lit,{value}\n"
        "dimly lit,{value}\n"
        "dramatically lit,{value}\n"
        "naturally lit,{value}\n"
        "warmly lit,{value}\n"
        "coolly lit,{value}\n"
        "backlit,{value}\n"
        "front lit,{value}\n"
        "side lit,{value}\n"
        "rim lit,{value}\n"
        "edge lit,{value}\n"
        "top lit,{value}\n"
        "under lit,{value}\n"
        "candlelit,{value}\n"
        "moonlit,{value}\n"
        "torchlit,{value}\n"
        "lantern lit,{value}\n"
        "firelit,{value}\n"
        "sunlit,{value}"
    )

    # ---- MOOD ---------------------------------------------------------------
    MOOD_OPTIONS = [
        "none",
        "intimate", "sensual", "cozy", "warm", "serene", "tranquil",
        "peaceful", "calm", "relaxed", "playful", "provocative", "cheerful",
        "inviting", "welcoming", "sophisticated", "elegant", "alluring",
        "mysterious", "confident", "professional", "casual", "comfortable",
        "contemplative", "dreamy", "vibrant", "festive", "moody", "airy",
        "low-key", "harmonious", "somber",
        "romantic", "whimsical", "nostalgic", "melancholic", "bittersweet",
        "wistful", "joyful", "uplifting", "heartwarming", "hopeful",
        "triumphant", "celebratory", "energetic", "lively", "tense",
        "anxious", "dramatic", "epic", "gritty", "edgy", "rebellious",
        "chaotic", "eerie", "ominous", "foreboding", "enigmatic",
        "introspective", "meditative", "spiritual", "reverent",
        "empowering", "vulnerable", "defiant", "curious", "mischievous",
    ]
    MOOD_MAPPING_DEFAULT = (
        "mood is intimate and sensual,mood is {value}\n"
        "mood is sensual and intimate,mood is {value}\n"
        "mood is relaxed and casual,mood is {value}\n"
        "mood is peaceful and serene,mood is {value}\n"
        "mood is calm and serene,mood is {value}\n"
        "mood is relaxed and intimate,mood is {value}\n"
        "mood is calm and contemplative,mood is {value}\n"
        "mood is playful and sensual,mood is {value}\n"
        "mood is casual and relaxed,mood is {value}\n"
        "mood is cheerful and inviting,mood is {value}\n"
        "mood is sensual and provocative,mood is {value}\n"
        "mood is playful and provocative,mood is {value}\n"
        "mood is serene and contemplative,mood is {value}\n"
        "mood is peaceful and contemplative,mood is {value}\n"
        "mood is relaxed and comfortable,mood is {value}\n"
        "mood is somber and contemplative,mood is {value}\n"
        "mood is warm and inviting,mood is {value}\n"
        "mood is sophisticated and elegant,mood is {value}\n"
        "mood is playful and alluring,mood is {value}\n"
        "mood is relaxed and confident,mood is {value}\n"
        "mood is professional and focused,mood is {value}\n"
        "mood is confident and professional,mood is {value}\n"
        "mood is cheerful and friendly,mood is {value}\n"
        "mood is calm and intimate,mood is {value}\n"
        "mood is mysterious and alluring,mood is {value}\n"
        "mood is playful and carefree,mood is {value}\n"
        "creating a warm and inviting atmosphere,creating a {value} atmosphere\n"
        "creating a cozy atmosphere,creating a {value} atmosphere\n"
        "creating a serene atmosphere,creating a {value} atmosphere\n"
        "creating a sensual atmosphere,creating a {value} atmosphere\n"
        "creating a warm and intimate atmosphere,creating a {value} atmosphere\n"
        "creating a peaceful atmosphere,creating a {value} atmosphere\n"
        "creating a bright and airy atmosphere,creating a {value} atmosphere\n"
        "creating a moody atmosphere,creating a {value} atmosphere\n"
        "creating a calming atmosphere,creating a {value} atmosphere\n"
        "creating a dreamy atmosphere,creating a {value} atmosphere\n"
        "creating a vibrant atmosphere,creating a {value} atmosphere\n"
        "creating a comfortable atmosphere,creating a {value} atmosphere\n"
        "creating a relaxed atmosphere,creating a {value} atmosphere\n"
        "warm and inviting atmosphere,{value} atmosphere\n"
        "cozy and intimate atmosphere,{value} atmosphere\n"
        "calm and intimate atmosphere,{value} atmosphere\n"
        "sensual and intimate atmosphere,{value} atmosphere\n"
        "bright and airy atmosphere,{value} atmosphere\n"
        "intimate,{value}\n"
        "sensual,{value}\n"
        "cozy,{value}\n"
        "serene,{value}\n"
        "tranquil,{value}\n"
        "playful,{value}\n"
        "provocative,{value}\n"
        "inviting,{value}\n"
        "welcoming,{value}\n"
        "sophisticated,{value}\n"
        "alluring,{value}\n"
        "mysterious,{value}\n"
        "dreamy,{value}\n"
        "vibrant,{value}\n"
        "festive,{value}\n"
        "romantic,{value}\n"
        "whimsical,{value}\n"
        "nostalgic,{value}\n"
        "melancholic,{value}\n"
        "bittersweet,{value}\n"
        "wistful,{value}\n"
        "uplifting,{value}\n"
        "heartwarming,{value}\n"
        "hopeful,{value}\n"
        "triumphant,{value}\n"
        "celebratory,{value}\n"
        "tense,{value}\n"
        "gritty,{value}\n"
        "edgy,{value}\n"
        "rebellious,{value}\n"
        "chaotic,{value}\n"
        "enigmatic,{value}\n"
        "introspective,{value}\n"
        "meditative,{value}\n"
        "reverent,{value}\n"
        "empowering,{value}\n"
        "vulnerable,{value}\n"
        "defiant,{value}"
    )

    # ---- POSE ---------------------------------------------------------------
    POSE_OPTIONS = [
        "none", "standing", "sitting", "lying down", "lying on back", 
        "lying on side", "lying on stomach", "kneeling", "on knees", 
        "one knee", "both knees", "crouching", "squatting", "bending over",
        "leaning forward", "leaning back", "leaning on one leg", 
        "standing on one leg", "one leg in the air", "leg up", 
        "crossed legs", "sitting cross-legged", "sitting on knees", 
        "sitting on the floor", "sitting on a chair", "sitting on a bench",
        "sitting sideways", "reclining", "lying on the ground",
        "lying on a bed", "lying on a couch", "on all fours",
        "hands and knees", "kneeling on one knee", "kneeling on both knees",
        "standing tall", "slouching", "hunched", "arched back",
        "twisted pose", "contrapposto", "dynamic pose", "action pose",
        "running", "walking", "jumping", "dancing", "stretching",
        "bending backwards", "bending forwards", "sideways bend",
        "split", "frog pose", "warrior pose", "yoga pose",
    ]
    POSE_MAPPING_DEFAULT = (
        "standing,{value}\n"
        "standing up,{value}\n"
        "standing pose,{value}\n"
        "standing tall,{value}\n"
        "standing straight,{value}\n"
        "standing casually,{value}\n"
        "standing relaxed,{value}\n"
        "sitting,{value}\n"
        "sitting down,{value}\n"
        "sitting pose,{value}\n"
        "seated,{value}\n"
        "sitting on a chair,{value}\n"
        "sitting on the floor,{value}\n"
        "sitting on the ground,{value}\n"
        "sitting cross-legged,{value}\n"
        "sitting on knees,{value}\n"
        "sitting sideways,{value}\n"
        "lying down,{value}\n"
        "lying on back,{value}\n"
        "lying on side,{value}\n"
        "lying on stomach,{value}\n"
        "lying on the ground,{value}\n"
        "lying on a bed,{value}\n"
        "lying on a couch,{value}\n"
        "reclining,{value}\n"
        "kneeling,{value}\n"
        "on knees,{value}\n"
        "on one knee,{value}\n"
        "on both knees,{value}\n"
        "kneeling on one knee,{value}\n"
        "kneeling on both knees,{value}\n"
        "crouching,{value}\n"
        "squatting,{value}\n"
        "bending over,{value}\n"
        "bent over,{value}\n"
        "bending forwards,{value}\n"
        "bending backwards,{value}\n"
        "leaning forward,{value}\n"
        "leaning backwards,{value}\n"
        "leaning back,{value}\n"
        "leaning on one leg,{value}\n"
        "standing on one leg,{value}\n"
        "one leg in the air,{value}\n"
        "leg up,{value}\n"
        "crossed legs,{value}\n"
        "on all fours,{value}\n"
        "hands and knees,{value}\n"
        "slouching,{value}\n"
        "hunched,{value}\n"
        "hunched over,{value}\n"
        "arched back,{value}\n"
        "contrapposto,{value}\n"
        "dynamic pose,{value}\n"
        "action pose,{value}\n"
        "running,{value}\n"
        "walking,{value}\n"
        "jumping,{value}\n"
        "dancing,{value}\n"
        "stretching,{value}\n"
        "split,{value}\n"
        "yoga pose,{value}"
    )

    # ---- HAND POSITION ------------------------------------------------------
    HAND_POSITION_OPTIONS = [
        "none", "hands at sides", "hands on hips", "arms crossed", 
        "hands clasped", "hands behind back", "hands in pockets", 
        "hands on face", "hands on head", "hands on shoulders",
        "hands on knees", "hands on thighs", "hands touching",
        "hands raised", "hands up", "one hand up", "both hands up",
        "hands waving", "hands gesturing", "hands reaching out",
        "hands reaching forward", "hands reaching up", "hands reaching down",
        "hands clasped together", "hands folded", "hands intertwined",
        "fingers interlaced", "pointing", "pointing finger", "pointing hand",
        "thumbs up", "peace sign", "hand on chin", "hand on cheek",
        "hand on hip", "hands on waist", "arms behind back",
        "arms in front", "arms outstretched", "arms open",
        "hands covering face", "hands covering mouth", "hands over eyes",
        "palms open", "palms up", "palms down", "fist", "clenched fist",
        "holding object", "holding something", "carrying", "lifting",
        "touching hair", "playing with hair", "adjusting hair",
        "hand in the air", "hands in the air", "arms raised",
        "hands on stomach", "hands on chest", "hands on heart",
    ]
    HAND_POSITION_MAPPING_DEFAULT = (
        "hands at sides,{value}\n"
        "hands at side,{value}\n"
        "hands on hips,{value}\n"
        "hand on hip,{value}\n"
        "hands on waist,{value}\n"
        "arms crossed,{value}\n"
        "crossed arms,{value}\n"
        "hands clasped,{value}\n"
        "hands clasped together,{value}\n"
        "hands behind back,{value}\n"
        "arms behind back,{value}\n"
        "hands in pockets,{value}\n"
        "hands in pocket,{value}\n"
        "hands on face,{value}\n"
        "hand on face,{value}\n"
        "hands on head,{value}\n"
        "hand on head,{value}\n"
        "hands on shoulders,{value}\n"
        "hands on knees,{value}\n"
        "hands on thighs,{value}\n"
        "hands touching,{value}\n"
        "hands raised,{value}\n"
        "hands up,{value}\n"
        "one hand up,{value}\n"
        "both hands up,{value}\n"
        "hands waving,{value}\n"
        "waving,{value}\n"
        "hands gesturing,{value}\n"
        "gesturing,{value}\n"
        "hands reaching out,{value}\n"
        "reaching out,{value}\n"
        "hands reaching forward,{value}\n"
        "reaching forward,{value}\n"
        "hands reaching up,{value}\n"
        "reaching up,{value}\n"
        "hands reaching down,{value}\n"
        "reaching down,{value}\n"
        "hands folded,{value}\n"
        "folded hands,{value}\n"
        "hands intertwined,{value}\n"
        "intertwined fingers,{value}\n"
        "fingers interlaced,{value}\n"
        "pointing,{value}\n"
        "pointing finger,{value}\n"
        "pointing hand,{value}\n"
        "thumbs up,{value}\n"
        "peace sign,{value}\n"
        "hand on chin,{value}\n"
        "hand on cheek,{value}\n"
        "arms outstretched,{value}\n"
        "outstretched arms,{value}\n"
        "arms open,{value}\n"
        "open arms,{value}\n"
        "hands covering face,{value}\n"
        "covering face,{value}\n"
        "hands covering mouth,{value}\n"
        "covering mouth,{value}\n"
        "hands over eyes,{value}\n"
        "palms open,{value}\n"
        "open palms,{value}\n"
        "palms up,{value}\n"
        "palms down,{value}\n"
        "fist,{value}\n"
        "clenched fist,{value}\n"
        "holding object,{value}\n"
        "holding something,{value}\n"
        "carrying,{value}\n"
        "lifting,{value}\n"
        "touching hair,{value}\n"
        "playing with hair,{value}\n"
        "adjusting hair,{value}\n"
        "hand in the air,{value}\n"
        "hands in the air,{value}\n"
        "arms raised,{value}\n"
        "raised arms,{value}\n"
        "hands on stomach,{value}\n"
        "hands on chest,{value}\n"
        "hands on heart,{value}"
    )

    # ---- HEAD POSITION ------------------------------------------------------
    HEAD_POSITION_OPTIONS = [
        "none", "looking forward", "looking straight ahead", "facing forward",
        "looking up", "looking down", "looking back", "looking over shoulder",
        "looking sideways", "looking left", "looking right",
        "head up", "head down", "head tilted", "head tilted left",
        "head tilted right", "head tilted up", "head tilted down",
        "head turned", "head turned left", "head turned right",
        "head turned back", "looking behind", "looking over shoulder",
        "chin up", "chin down", "chin raised", "chin tucked",
        "face up", "face down", "facing up", "facing down",
        "facing away", "facing towards", "profile view", "in profile",
        "looking at camera", "looking at viewer", "looking away",
        "eyes looking up", "eyes looking down", "eyes looking sideways",
        "looking off-screen", "looking into the distance",
        "gazing up", "gazing down", "gazing sideways",
        "head resting on hands", "head in hands", 
        "head leaning on hand", "head resting on chin",
        "tilted head", "cocked head", "head cocked",
    ]
    HEAD_POSITION_MAPPING_DEFAULT = (
        "looking forward,{value}\n"
        "looking straight ahead,{value}\n"
        "straight ahead,{value}\n"
        "facing forward,{value}\n"
        "looking up,{value}\n"
        "looking down,{value}\n"
        "looking back,{value}\n"
        "looking over shoulder,{value}\n"
        "over shoulder,{value}\n"
        "looking sideways,{value}\n"
        "looking left,{value}\n"
        "looking right,{value}\n"
        "head up,{value}\n"
        "head down,{value}\n"
        "head tilted,{value}\n"
        "head tilted left,{value}\n"
        "head tilted right,{value}\n"
        "head turned,{value}\n"
        "head turned left,{value}\n"
        "head turned right,{value}\n"
        "head turned back,{value}\n"
        "looking behind,{value}\n"
        "chin up,{value}\n"
        "chin down,{value}\n"
        "chin raised,{value}\n"
        "chin tucked,{value}\n"
        "face up,{value}\n"
        "face down,{value}\n"
        "facing up,{value}\n"
        "facing down,{value}\n"
        "facing away,{value}\n"
        "facing towards,{value}\n"
        "profile view,{value}\n"
        "in profile,{value}\n"
        "looking at camera,{value}\n"
        "looking at viewer,{value}\n"
        "looking away,{value}\n"
        "eyes looking up,{value}\n"
        "eyes looking down,{value}\n"
        "eyes looking sideways,{value}\n"
        "looking off-screen,{value}\n"
        "looking into the distance,{value}\n"
        "gazing up,{value}\n"
        "gazing down,{value}\n"
        "gazing sideways,{value}\n"
        "head resting on hands,{value}\n"
        "head in hands,{value}\n"
        "head leaning on hand,{value}\n"
        "head resting on chin,{value}\n"
        "tilted head,{value}\n"
        "cocked head,{value}\n"
        "head cocked,{value}"
    )

    # ---- TOP COLOR ----------------------------------------------------------
    TOP_COLOR_OPTIONS = [
        "none", "black", "white", "red", "blue", "green", "yellow", "purple",
        "pink", "orange", "brown", "gray", "navy", "teal", "turquoise",
        "magenta", "coral", "maroon", "olive", "cream", "beige", "gold",
        "silver", "pastel pink", "pastel blue", "pastel yellow", "pastel green",
        "bright red", "bright blue", "bright yellow", "dark red", "dark blue",
        "dark green", "light blue", "light pink", "light gray", "charcoal",
        "burgundy", "mustard", "mint green", "lavender", "peach",
    ]
    TOP_COLOR_MAPPING_DEFAULT = (
        "black shirt,{value} shirt\n"
        "white shirt,{value} shirt\n"
        "red shirt,{value} shirt\n"
        "blue shirt,{value} shirt\n"
        "green shirt,{value} shirt\n"
        "yellow shirt,{value} shirt\n"
        "purple shirt,{value} shirt\n"
        "pink shirt,{value} shirt\n"
        "orange shirt,{value} shirt\n"
        "brown shirt,{value} shirt\n"
        "gray shirt,{value} shirt\n"
        "black top,{value} top\n"
        "white top,{value} top\n"
        "red top,{value} top\n"
        "blue top,{value} top\n"
        "green top,{value} top\n"
        "black blouse,{value} blouse\n"
        "white blouse,{value} blouse\n"
        "red blouse,{value} blouse\n"
        "blue blouse,{value} blouse\n"
        "black t-shirt,{value} t-shirt\n"
        "white t-shirt,{value} t-shirt\n"
        "black sweater,{value} sweater\n"
        "white sweater,{value} sweater\n"
        "black jacket,{value} jacket\n"
        "white jacket,{value} jacket\n"
        "black hoodie,{value} hoodie\n"
        "white hoodie,{value} hoodie\n"
        "black dress,{value} dress\n"
        "white dress,{value} dress\n"
        "red dress,{value} dress\n"
        "blue dress,{value} dress\n"
        "wearing black, wearing {value}\n"
        "wearing white, wearing {value}\n"
        "wearing red, wearing {value}\n"
        "wearing blue, wearing {value}\n"
        "dressed in black, dressed in {value}\n"
        "dressed in white, dressed in {value}\n"
        "dressed in red, dressed in {value}\n"
        "dressed in blue, dressed in {value}\n"
    )

    # ---- TOP TYPE -----------------------------------------------------------
    TOP_TYPE_OPTIONS = [
        "none", "t-shirt", "shirt", "blouse", "sweater", "jumper", "cardigan",
        "jacket", "hoodie", "coat", "vest", "tank top", "crop top",
        "tube top", "halter top", "off-shoulder", "one-shoulder",
        "dress", "sundress", "evening gown", "formal shirt",
        "button-up shirt", "collared shirt", "v-neck", "crew neck",
        "turtleneck", "scoop neck", "sweatshirt", "blazer",
        "leather jacket", "denim jacket", "bomber jacket",
        "raincoat", "parka", "windbreaker", "polo shirt",
    ]
    TOP_TYPE_MAPPING_DEFAULT = (
        "t-shirt,{value}\n"
        "tshirt,{value}\n"
        "tee,{value}\n"
        "shirt,{value}\n"
        "blouse,{value}\n"
        "sweater,{value}\n"
        "jumper,{value}\n"
        "cardigan,{value}\n"
        "jacket,{value}\n"
        "hoodie,{value}\n"
        "coat,{value}\n"
        "vest,{value}\n"
        "tank top,{value}\n"
        "crop top,{value}\n"
        "tube top,{value}\n"
        "halter top,{value}\n"
        "off-shoulder,{value}\n"
        "one-shoulder,{value}\n"
        "dress,{value}\n"
        "sundress,{value}\n"
        "evening gown,{value}\n"
        "gown,{value}\n"
        "button-up shirt,{value}\n"
        "button down,{value}\n"
        "collared shirt,{value}\n"
        "v-neck,{value}\n"
        "crew neck,{value}\n"
        "turtleneck,{value}\n"
        "sweatshirt,{value}\n"
        "blazer,{value}\n"
        "leather jacket,{value}\n"
        "denim jacket,{value}\n"
        "bomber jacket,{value}\n"
        "raincoat,{value}\n"
        "parka,{value}\n"
        "windbreaker,{value}\n"
        "polo shirt,{value}\n"
        "top,{value}\n"
        "upper body,{value}\n"
    )

    # ---- BOTTOM COLOR -------------------------------------------------------
    BOTTOM_COLOR_OPTIONS = [
        "none", "black", "white", "blue", "gray", "brown", "beige", "cream",
        "navy", "dark blue", "light blue", "khaki", "olive", "tan",
        "red", "green", "purple", "pink", "yellow", "orange",
        "charcoal", "denim blue", "light wash", "dark wash",
        "pastel", "bright", "neon",
    ]
    BOTTOM_COLOR_MAPPING_DEFAULT = (
        "black pants,{value} pants\n"
        "white pants,{value} pants\n"
        "blue pants,{value} pants\n"
        "gray pants,{value} pants\n"
        "brown pants,{value} pants\n"
        "black jeans,{value} jeans\n"
        "blue jeans,{value} jeans\n"
        "gray jeans,{value} jeans\n"
        "black skirt,{value} skirt\n"
        "white skirt,{value} skirt\n"
        "blue skirt,{value} skirt\n"
        "gray skirt,{value} skirt\n"
        "black shorts,{value} shorts\n"
        "blue shorts,{value} shorts\n"
        "gray shorts,{value} shorts\n"
        "black trousers,{value} trousers\n"
        "gray trousers,{value} trousers\n"
        "black leggings,{value} leggings\n"
        "gray leggings,{value} leggings\n"
        "black bottoms,{value} bottoms\n"
        "blue bottoms,{value} bottoms\n"
        "wearing black pants, wearing {value} pants\n"
        "wearing blue jeans, wearing {value} jeans\n"
        "wearing black skirt, wearing {value} skirt\n"
        "wearing black shorts, wearing {value} shorts\n"
    )

    # ---- BOTTOM TYPE --------------------------------------------------------
    BOTTOM_TYPE_OPTIONS = [
        "none", "pants", "trousers", "jeans", "skinny jeans", "relaxed jeans",
        "wide-leg pants", "cargo pants", "sweatpants", "joggers",
        "shorts", "bermuda shorts", "denim shorts", "skirt", "mini skirt",
        "midi skirt", "maxi skirt", "pleated skirt", "pencil skirt",
        "leggings", "tights", "stockings", "socks",
        "bottoms", "underwear", "panties", "boxers", "briefs",
        "swim trunks", "swim shorts", "bikini bottoms",
    ]
    BOTTOM_TYPE_MAPPING_DEFAULT = (
        "pants,{value}\n"
        "trousers,{value}\n"
        "jeans,{value}\n"
        "skinny jeans,{value}\n"
        "relaxed jeans,{value}\n"
        "wide-leg pants,{value}\n"
        "cargo pants,{value}\n"
        "sweatpants,{value}\n"
        "joggers,{value}\n"
        "shorts,{value}\n"
        "bermuda shorts,{value}\n"
        "denim shorts,{value}\n"
        "skirt,{value}\n"
        "mini skirt,{value}\n"
        "midi skirt,{value}\n"
        "maxi skirt,{value}\n"
        "pleated skirt,{value}\n"
        "pencil skirt,{value}\n"
        "leggings,{value}\n"
        "tights,{value}\n"
        "stockings,{value}\n"
        "bottoms,{value}\n"
        "underwear,{value}\n"
        "swim trunks,{value}\n"
        "swim shorts,{value}\n"
        "bikini bottoms,{value}\n"
    )

    # ---- FOOTWEAR COLOR -----------------------------------------------------
    FOOTWEAR_COLOR_OPTIONS = [
        "none", "black", "white", "brown", "tan", "beige", "gray",
        "blue", "red", "green", "purple", "pink", "yellow", "orange",
        "navy", "dark brown", "light brown", "charcoal", "silver",
        "gold", "burgundy", "olive", "cream", "multicolor",
    ]
    FOOTWEAR_COLOR_MAPPING_DEFAULT = (
        "black shoes,{value} shoes\n"
        "white shoes,{value} shoes\n"
        "brown shoes,{value} shoes\n"
        "gray shoes,{value} shoes\n"
        "blue shoes,{value} shoes\n"
        "red shoes,{value} shoes\n"
        "black boots,{value} boots\n"
        "brown boots,{value} boots\n"
        "white boots,{value} boots\n"
        "black sneakers,{value} sneakers\n"
        "white sneakers,{value} sneakers\n"
        "black trainers,{value} trainers\n"
        "white trainers,{value} trainers\n"
        "black sandals,{value} sandals\n"
        "brown sandals,{value} sandals\n"
        "black heels,{value} heels\n"
        "black footwear,{value} footwear\n"
        "wearing black shoes, wearing {value} shoes\n"
        "wearing white sneakers, wearing {value} sneakers\n"
    )

    # ---- FOOTWEAR TYPE ------------------------------------------------------
    FOOTWEAR_TYPE_OPTIONS = [
        "none", "shoes", "boots", "ankle boots", "knee-high boots",
        "sneakers", "trainers", "runners", "tennis shoes", "basketball shoes",
        "sandals", "flip-flops", "slides", "heels", "stilettos", "wedges",
        "loafers", "oxfords", "broguers", "derbies", "ballet flats",
        "flats", "mules", "clogs", "slippers", "house shoes",
        "high-tops", "low-tops", "chunky sneakers", "platform shoes",
        "combat boots", "work boots", "hiking boots", "winter boots",
        "rain boots", "wellies", "cowboy boots", "riding boots",
        "pumps", "kitten heels", "block heels", "stiletto heels",
        "espadrilles", "canvas shoes", "leather shoes", "suede shoes",
        "barefoot", "no shoes",
    ]
    FOOTWEAR_TYPE_MAPPING_DEFAULT = (
        "shoes,{value}\n"
        "boots,{value}\n"
        "ankle boots,{value}\n"
        "knee-high boots,{value}\n"
        "sneakers,{value}\n"
        "trainers,{value}\n"
        "runners,{value}\n"
        "tennis shoes,{value}\n"
        "sandals,{value}\n"
        "flip-flops,{value}\n"
        "heels,{value}\n"
        "stilettos,{value}\n"
        "wedges,{value}\n"
        "loafers,{value}\n"
        "oxfords,{value}\n"
        "ballet flats,{value}\n"
        "flats,{value}\n"
        "mules,{value}\n"
        "clogs,{value}\n"
        "slippers,{value}\n"
        "high-tops,{value}\n"
        "chunky sneakers,{value}\n"
        "platform shoes,{value}\n"
        "combat boots,{value}\n"
        "work boots,{value}\n"
        "hiking boots,{value}\n"
        "winter boots,{value}\n"
        "rain boots,{value}\n"
        "cowboy boots,{value}\n"
        "riding boots,{value}\n"
        "pumps,{value}\n"
        "kitten heels,{value}\n"
        "block heels,{value}\n"
        "stiletto heels,{value}\n"
        "espadrilles,{value}\n"
        "canvas shoes,{value}\n"
        "leather shoes,{value}\n"
        "suede shoes,{value}\n"
        "barefoot,{value}\n"
        "no shoes,{value}\n"
        "footwear,{value}\n"
    )

    # ---- OUTFIT STYLE -------------------------------------------------------
    OUTFIT_STYLE_OPTIONS = [
        "none", "casual", "formal", "business", "business casual",
        "smart casual", "sporty", "athletic", "elegant", "glamorous",
        "bohemian", "vintage", "retro", "grunge", "punk", "gothic",
        "minimalist", "streetwear", "hip-hop", "preppy", "classic",
        "romantic", "sexy", "revealing", "modest", "conservative",
        "summer", "spring", "autumn", "winter", "rainy day", "beachwear",
        "swimwear", "lingerie", "nightwear", "loungewear",
        "suit", "tuxedo", "evening gown", "wedding dress", "prom dress",
        "costume", "cosplay", "uniform", "armor", "robe", "kimono",
    ]
    OUTFIT_STYLE_MAPPING_DEFAULT = (
        "casual outfit,{value}\n"
        "casual,{value}\n"
        "formal wear,{value}\n"
        "formal,{value}\n"
        "business casual,{value}\n"
        "business attire,{value}\n"
        "sporty,{value}\n"
        "athletic wear,{value}\n"
        "elegant,{value}\n"
        "glamorous,{value}\n"
        "bohemian,{value}\n"
        "vintage,{value}\n"
        "retro style,{value}\n"
        "grunge,{value}\n"
        "punk,{value}\n"
        "gothic,{value}\n"
        "minimalist,{value}\n"
        "streetwear,{value}\n"
        "preppy,{value}\n"
        "classic style,{value}\n"
        "romantic,{value}\n"
        "sexy,{value}\n"
        "revealing,{value}\n"
        "modest,{value}\n"
        "summer outfit,{value}\n"
        "winter outfit,{value}\n"
        "beachwear,{value}\n"
        "swimwear,{value}\n"
        "lingerie,{value}\n"
        "loungewear,{value}\n"
        "suit,{value}\n"
        "tuxedo,{value}\n"
        "evening gown,{value}\n"
        "wedding dress,{value}\n"
        "costume,{value}\n"
        "cosplay,{value}\n"
        "uniform,{value}\n"
        "robe,{value}\n"
        "kimono,{value}\n"
        "outfit style,{value}\n"
        "style,{value}\n"
    )

    # ---- CAMERA ANGLE -------------------------------------------------------
    CAMERA_ANGLE_OPTIONS = [
        "none",
        "low angle", "slightly elevated angle", "high angle", "selfie angle",
        "front angle", "close-up angle", "side angle", "portrait angle",
        "profile angle",
        "close-up", "extreme close-up", "medium shot", "medium close-up",
        "full body shot", "wide shot", "extreme wide shot",
        "establishing shot", "eye level", "birds eye view", "worms eye view",
        "dutch angle", "over the shoulder", "point of view",
        "three quarter angle", "front facing", "profile view", "back view",
        "aerial view", "overhead shot", "top down view", "tilted angle",
        "macro shot", "telephoto shot", "wide angle lens", "fisheye lens",
        "panoramic", "tracking shot", "dolly shot", "crane shot",
        "handheld shot", "drone shot",
    ]
    CAMERA_ANGLE_MAPPING_DEFAULT = (
        "taken from a low angle,taken from a {value}\n"
        "taken from a slightly elevated angle,taken from a {value}\n"
        "taken from a high angle,taken from a {value}\n"
        "taken from a slightly low angle,taken from a {value}\n"
        "taken from a slightly lower angle,taken from a {value}\n"
        "taken from a selfie angle,taken from a {value}\n"
        "front camera angle,{value}\n"
        "close-up camera angle,{value}\n"
        "close up camera angle,{value}\n"
        "side camera angle,{value}\n"
        "portrait camera angle,{value}\n"
        "profile camera angle,{value}\n"
        "video camera angle,{value}\n"
        "frontal camera angle,{value}\n"
        "full body camera angle,{value}\n"
        "selfie camera angle,{value}\n"
        "close-up shot,{value}\n"
        "extreme close-up,{value}\n"
        "medium shot,{value}\n"
        "medium close-up,{value}\n"
        "full body shot,{value}\n"
        "wide shot,{value}\n"
        "extreme wide shot,{value}\n"
        "establishing shot,{value}\n"
        "low angle shot,{value}\n"
        "high angle shot,{value}\n"
        "eye level shot,{value}\n"
        "eye level,{value}\n"
        "bird's eye view,{value}\n"
        "birds eye view,{value}\n"
        "worm's eye view,{value}\n"
        "worms eye view,{value}\n"
        "dutch angle,{value}\n"
        "over-the-shoulder shot,{value}\n"
        "over the shoulder shot,{value}\n"
        "point of view shot,{value}\n"
        "pov shot,{value}\n"
        "three-quarter angle,{value}\n"
        "three quarter angle,{value}\n"
        "front-facing shot,{value}\n"
        "back view,{value}\n"
        "aerial view,{value}\n"
        "overhead shot,{value}\n"
        "top-down view,{value}\n"
        "top down view,{value}\n"
        "drone shot,{value}\n"
        "tilted angle,{value}\n"
        "macro shot,{value}\n"
        "telephoto shot,{value}\n"
        "wide-angle lens,{value} lens\n"
        "fisheye lens,{value} lens\n"
        "panoramic shot,{value}\n"
        "tracking shot,{value}\n"
        "dolly shot,{value}\n"
        "crane shot,{value}\n"
        "handheld shot,{value}"
    )

    # ---- SETTING ------------------------------------------------------------
    SETTING_OPTIONS = [
        "none",
        "living room", "bedroom", "studio", "kitchen", "garden", "park",
        "gym", "indoor", "outdoor",
        "bathroom", "home office", "photo studio", "beach", "forest",
        "mountain", "city street", "rooftop", "balcony", "cafe",
        "coffee shop", "restaurant", "bar", "nightclub", "office",
        "boardroom", "classroom", "library", "museum", "art gallery",
        "hotel room", "hotel lobby", "poolside", "backyard", "patio",
        "greenhouse", "countryside", "desert", "urban alley",
        "subway station", "train station", "airport", "staircase",
        "hallway", "rooftop terrace", "courtyard", "church", "cathedral",
        "castle", "warehouse", "abandoned building", "forest clearing",
        "snowy landscape", "underwater", "spaceship", "futuristic city",
    ]
    SETTING_MAPPING_DEFAULT = (
        "modern living room,{value}\n"
        "cozy living room,{value}\n"
        "minimalistic living room,{value}\n"
        "modern kitchen,{value}\n"
        "modern bedroom,{value}\n"
        "cozy bedroom,{value}\n"
        "modern interior,{value}\n"
        "minimalistic indoor,{value}\n"
        "living room,{value}\n"
        "bedroom,{value}\n"
        "studio,{value}\n"
        "garden,{value}\n"
        "park,{value}\n"
        "gym,{value}\n"
        "kitchen,{value}\n"
        "bathroom,{value}\n"
        "home office,{value}\n"
        "photo studio,{value}\n"
        "outdoor garden,{value}\n"
        "beach,{value}\n"
        "forest,{value}\n"
        "mountain,{value}\n"
        "city street,{value}\n"
        "rooftop,{value}\n"
        "balcony,{value}\n"
        "cafe,{value}\n"
        "coffee shop,{value}\n"
        "restaurant,{value}\n"
        "bar,{value}\n"
        "nightclub,{value}\n"
        "office,{value}\n"
        "boardroom,{value}\n"
        "classroom,{value}\n"
        "library,{value}\n"
        "museum,{value}\n"
        "art gallery,{value}\n"
        "hotel room,{value}\n"
        "hotel lobby,{value}\n"
        "poolside,{value}\n"
        "backyard,{value}\n"
        "patio,{value}\n"
        "greenhouse,{value}\n"
        "countryside,{value}\n"
        "desert,{value}\n"
        "urban alley,{value}\n"
        "subway station,{value}\n"
        "train station,{value}\n"
        "airport,{value}\n"
        "staircase,{value}\n"
        "hallway,{value}\n"
        "rooftop terrace,{value}\n"
        "courtyard,{value}\n"
        "church,{value}\n"
        "cathedral,{value}\n"
        "castle,{value}\n"
        "warehouse,{value}\n"
        "abandoned building,{value}\n"
        "forest clearing,{value}\n"
        "snowy landscape,{value}\n"
        "underwater,{value}\n"
        "spaceship,{value}\n"
        "futuristic city,{value}"
    )

    # ---- MAKEUP -------------------------------------------------------------
    MAKEUP_OPTIONS = [
        "none",
        "no makeup", "minimal makeup", "light makeup", "natural makeup",
        "subtle smokey eye", "dramatic eye makeup", "dark eye makeup",
        "subtle makeup", "glam makeup", "full glam makeup", "bold makeup",
        "smokey eye", "winged eyeliner", "cat eye makeup", "glossy lips",
        "matte lips", "red lipstick", "nude lipstick", "bold lipstick",
        "contoured makeup", "dewy makeup", "matte finish makeup",
        "editorial makeup", "vintage makeup", "gothic makeup",
        "sun-kissed makeup", "bronzed makeup", "runway makeup",
    ]
    MAKEUP_MAPPING_DEFAULT = (
        "is wearing minimal makeup,is wearing {value}\n"
        "wearing minimal makeup,wearing {value}\n"
        "is wearing light makeup,is wearing {value}\n"
        "wearing light makeup,wearing {value}\n"
        "natural-looking makeup,{value}\n"
        "subtle smokey eye,{value}\n"
        "subtle smoky eye,{value}\n"
        "smoky eye,{value}\n"
        "smokey eye,{value}\n"
        "dramatic eye makeup,{value}\n"
        "dark eye makeup,{value}\n"
        "no makeup,{value}\n"
        "minimal makeup,{value}\n"
        "light makeup,{value}\n"
        "natural makeup,{value}\n"
        "subtle makeup,{value}\n"
        "glam makeup,{value}\n"
        "full glam makeup,{value}\n"
        "dramatic makeup,{value}\n"
        "bold makeup,{value}\n"
        "winged eyeliner,{value}\n"
        "cat eye makeup,{value}\n"
        "glossy lips,{value}\n"
        "matte lips,{value}\n"
        "red lipstick,{value}\n"
        "nude lipstick,{value}\n"
        "bold lipstick,{value}\n"
        "contoured makeup,{value}\n"
        "dewy makeup,{value}\n"
        "matte makeup,{value}\n"
        "editorial makeup,{value}\n"
        "vintage makeup,{value}\n"
        "gothic makeup,{value}\n"
        "sun-kissed makeup,{value}\n"
        "bronzed makeup,{value}\n"
        "runway makeup,{value}"
    )

    # ---- JEWELRY ------------------------------------------------------------
    JEWELRY_OPTIONS = [
        "none",
        "gold hoop earrings", "diamond stud earrings", "gold necklace",
        "silver necklace", "pearl necklace", "gold chain necklace",
        "delicate gold necklace", "delicate silver necklace",
        "no jewelry", "gold jewelry", "silver jewelry", "rose gold jewelry",
        "platinum jewelry", "diamond jewelry", "pearl jewelry",
        "gemstone jewelry", "beaded jewelry", "minimalist jewelry",
        "statement jewelry", "layered necklaces", "chunky jewelry",
        "delicate jewelry", "vintage jewelry", "bohemian jewelry",
        "pearl earrings", "dangling earrings", "choker necklace",
        "pendant necklace", "charm bracelet", "bangle bracelet",
        "cuff bracelet", "statement ring", "diamond ring", "wedding ring",
        "engagement ring",
    ]
    JEWELRY_MAPPING_DEFAULT = (
        "gold hoop earrings,{value}\n"
        "large gold hoop earrings,{value}\n"
        "diamond stud earrings,{value}\n"
        "pair of diamond stud earrings,{value}\n"
        "gold stud earrings,{value}\n"
        "pair of gold earrings,{value}\n"
        "dangling earrings,{value}\n"
        "pair of dangling earrings,{value}\n"
        "silver necklace,{value}\n"
        "gold necklace,{value}\n"
        "pearl necklace,{value}\n"
        "gold chain,{value}\n"
        "gold chain necklace,{value}\n"
        "delicate gold necklace,{value}\n"
        "delicate silver necklace,{value}\n"
        "thin gold necklace,{value}\n"
        "gold-colored necklace,{value}\n"
        "no jewelry,{value}\n"
        "minimalist jewelry,{value}\n"
        "statement jewelry,{value}\n"
        "layered necklaces,{value}\n"
        "chunky jewelry,{value}\n"
        "delicate jewelry,{value}\n"
        "vintage jewelry,{value}\n"
        "bohemian jewelry,{value}\n"
        "pearl earrings,{value}\n"
        "choker necklace,{value}\n"
        "pendant necklace,{value}\n"
        "charm bracelet,{value}\n"
        "bangle bracelet,{value}\n"
        "cuff bracelet,{value}\n"
        "statement ring,{value}\n"
        "diamond ring,{value}\n"
        "wedding ring,{value}\n"
        "engagement ring,{value}"
    )

    # ---- CATEGORIES dictionary ---------------------------------------------
    CATEGORIES = {
        "photograph_type": (PHOTOGRAPH_TYPE_OPTIONS, PHOTOGRAPH_TYPE_MAPPING_DEFAULT),
        "setting": (SETTING_OPTIONS, SETTING_MAPPING_DEFAULT),
        "lighting": (LIGHTING_OPTIONS, LIGHTING_MAPPING_DEFAULT),
        "mood": (MOOD_OPTIONS, MOOD_MAPPING_DEFAULT),
        "camera_angle": (CAMERA_ANGLE_OPTIONS, CAMERA_ANGLE_MAPPING_DEFAULT),
        "hair_color": (HAIR_COLOR_OPTIONS, HAIR_COLOR_MAPPING_DEFAULT),
        "skin_tone": (SKIN_TONE_OPTIONS, SKIN_TONE_MAPPING_DEFAULT),
        "hair_style": (HAIR_STYLE_OPTIONS, HAIR_STYLE_MAPPING_DEFAULT),
        "eye_color": (EYE_COLOR_OPTIONS, EYE_COLOR_MAPPING_DEFAULT),
        "smile": (SMILE_OPTIONS, SMILE_MAPPING_DEFAULT),
        "expression": (EXPRESSION_OPTIONS, EXPRESSION_MAPPING_DEFAULT),
        "pose": (POSE_OPTIONS, POSE_MAPPING_DEFAULT),
        "outfit_style": (OUTFIT_STYLE_OPTIONS, OUTFIT_STYLE_MAPPING_DEFAULT),
        "headwear": (HEADWEAR_OPTIONS, HEADWEAR_MAPPING_DEFAULT),
        "hand_position": (HAND_POSITION_OPTIONS, HAND_POSITION_MAPPING_DEFAULT),
        "head_position": (HEAD_POSITION_OPTIONS, HEAD_POSITION_MAPPING_DEFAULT),
        "makeup": (MAKEUP_OPTIONS, MAKEUP_MAPPING_DEFAULT),
        "jewelry": (JEWELRY_OPTIONS, JEWELRY_MAPPING_DEFAULT),
        "top_color": (TOP_COLOR_OPTIONS, TOP_COLOR_MAPPING_DEFAULT),
        "top_type": (TOP_TYPE_OPTIONS, TOP_TYPE_MAPPING_DEFAULT),
        "bottom_color": (BOTTOM_COLOR_OPTIONS, BOTTOM_COLOR_MAPPING_DEFAULT),
        "bottom_type": (BOTTOM_TYPE_OPTIONS, BOTTOM_TYPE_MAPPING_DEFAULT),
        "footwear_color": (FOOTWEAR_COLOR_OPTIONS, FOOTWEAR_COLOR_MAPPING_DEFAULT),
        "footwear_type": (FOOTWEAR_TYPE_OPTIONS, FOOTWEAR_TYPE_MAPPING_DEFAULT),
    }

    # Categories that need special handling
    SPECIAL_CATEGORIES = {
        "photograph_type": "photograph_fallback",
    }

    # List of attribute categories for randomization
    RANDOMIZABLE_CATEGORIES = [
        "hair_color", "skin_tone", "hair_style", "eye_color", "smile", 
        "expression", "lighting", "mood", "pose", "hand_position",
        "head_position", "camera_angle", "setting", "makeup", "jewelry",
        "top_color", "top_type", "bottom_color", "bottom_type",
        "footwear_color", "footwear_type", "outfit_style",
        "photograph_type", "headwear",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for cat_key, (options, mapping_default) in cls.CATEGORIES.items():
            label = cat_key.replace("_", " ").title()
            tooltip = None
            if cat_key == "photograph_type":
                tooltip = "Intelligently inserts distance/shot type (e.g., 'close up photograph')"
            
            optional[cat_key] = (options, {
                "default": "none", 
                "label": label,
                "tooltip": tooltip
            })

        optional.update({
            "case_sensitive": ("BOOLEAN", {
                "default": False,
                "label": "Case Sensitive Matching",
            }),
            "match_whole_words": ("BOOLEAN", {
                "default": True,
                "label": "Match Whole Words Only",
                "tooltip": "When enabled, 'man' won't match 'woman'",
            }),
            "sort_by_length": ("BOOLEAN", {
                "default": True,
                "label": "Sort Rules by Length (Longest First)",
                "tooltip": "Prevents shorter phrases from replacing parts of longer phrases",
            }),
            "preserve_case": ("BOOLEAN", {
                "default": True,
                "label": "Preserve Original Case Pattern",
                "tooltip": "Attempt to maintain the case pattern of the original text",
            }),
            "highlight_format": (["markdown", "html", "plain"], {
                "default": "markdown",
                "label": "Highlight Format",
            }),
            "randomize": ("BOOLEAN", {
                "default": False,
                "label": "Randomize Attributes",
                "tooltip": "When enabled, randomly selects values for attributes set to 'none'",
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFF,
                "label": "Seed",
                "tooltip": "Seed for reproducible randomization (0 = random seed)",
            }),
        })

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter text to process...",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "highlighted_text", "rules_applied")
    FUNCTION = "replace_attributes"
    CATEGORY = "GR Utilities"

    # ---- Helper methods for special categories --------------------------------

    def apply_photograph_type_fallback(self, text, photograph_type_value, preserve_case=True):
        """
        Intelligently handle photograph type with fallback insertion.
        If no photograph descriptor exists, insert one appropriately.
        """
        if not text or not photograph_type_value or photograph_type_value == "none":
            return text, []

        changes = []
        
        # Check if any photograph-related terms exist in the text
        photo_terms = ['photograph', 'photo', 'image', 'picture', 'shot', 'selfie']
        has_photo_term = any(term in text.lower() for term in photo_terms)
        
        # Check if any distance/shot descriptor already exists
        distance_terms = ['close up', 'close-up', 'extreme close up', 'distant', 'wide', 'long', 
                         'full', 'medium', 'portrait', 'headshot', 'bust', 'candid', 'action',
                         'low angle', 'high angle', 'eye level', 'birds eye', 'worm\'s eye',
                         'overhead', 'aerial', 'drone', 'selfie']
        has_distance_desc = any(term in text.lower() for term in distance_terms)
        
        if has_photo_term and not has_distance_desc:
            # Need to insert the photograph type
            # Find where the photo term is and insert before it
            photo_pattern = r'\b(photograph|photo|image|picture|shot)\b'
            match = re.search(photo_pattern, text, re.IGNORECASE)
            
            if match:
                # Insert the photograph type before the photo term
                pos = match.start()
                new_text = text[:pos] + photograph_type_value + " " + text[pos:]
                
                changes.append({
                    'old': '',
                    'new': photograph_type_value + " ",
                    'start': pos,
                    'end': pos,
                    'rule_used': f"Insert photograph type: {photograph_type_value}"
                })
                return new_text, changes
            else:
                # Fallback: just prepend
                new_text = photograph_type_value + " " + text
                changes.append({
                    'old': '',
                    'new': photograph_type_value + " ",
                    'start': 0,
                    'end': 0,
                    'rule_used': f"Prepend photograph type: {photograph_type_value}"
                })
                return new_text, changes
        elif not has_photo_term:
            # No photograph term at all - insert "photograph" with the type
            # Find where to insert naturally
            # Common patterns: starts with article, or just prepend
            text_lower = text.lower()
            article_match = re.match(r'^(a|an|the)\s+', text_lower)
            
            if article_match:
                # Insert after the article
                pos = article_match.end()
                new_text = text[:pos] + photograph_type_value + " photograph " + text[pos:]
                changes.append({
                    'old': text[:pos],
                    'new': text[:pos] + photograph_type_value + " photograph ",
                    'start': 0,
                    'end': pos,
                    'rule_used': f"Insert photograph with type: {photograph_type_value}"
                })
                return new_text, changes
            else:
                # Prepend with "a" or "an"
                first_word = photograph_type_value.lower()
                if first_word[0] in 'aeiou':
                    article = 'an'
                else:
                    article = 'a'
                
                prefix = f"{article} {photograph_type_value} photograph of"
                new_text = prefix + " " + text
                changes.append({
                    'old': '',
                    'new': prefix + " ",
                    'start': 0,
                    'end': 0,
                    'rule_used': f"Prepend photograph with type: {photograph_type_value}"
                })
                return new_text, changes
        
        # If both exist or no photo term found, return unchanged
        return text, []

    # ---- Self-contained replacement engine -----------------------------------

    def parse_rules(self, rules_text):
        """Parse the multi-line replacement rules into a list of (old, new) tuples."""
        rules = []
        lines = rules_text.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = [part.strip() for part in line.split(',', 1)]

            if len(parts) == 2:
                old_phrase, new_phrase = parts
                if old_phrase:
                    old_phrase = ' '.join(old_phrase.split())
                    new_phrase = ' '.join(new_phrase.split())
                    rules.append((old_phrase, new_phrase))
            else:
                print(f"Warning: Line {line_num} in replacement rules is malformed: '{line}'")

        return rules

    def escape_for_regex(self, text):
        """Escape text for regex but preserve word boundaries for phrases."""
        escaped = re.escape(text)
        escaped = escaped.replace(r'\ ', r'\s+')
        return escaped

    def create_whole_word_pattern(self, phrase):
        """Create a pattern that matches whole words, handling multi-word phrases."""
        words = phrase.split()
        if len(words) == 1:
            return r'\b' + re.escape(words[0]) + r'\b'
        else:
            pattern_parts = []
            for i, word in enumerate(words):
                escaped = re.escape(word)
                if i == 0:
                    pattern_parts.append(r'\b' + escaped)
                elif i == len(words) - 1:
                    pattern_parts.append(escaped + r'\b')
                else:
                    pattern_parts.append(escaped)
            return r'\s+'.join(pattern_parts)

    def preserve_case_pattern(self, matched_text, replacement):
        """Attempt to preserve the case pattern of the matched text."""
        if not matched_text or not replacement:
            return replacement

        if matched_text.isupper():
            return replacement.upper()
        elif matched_text.islower():
            return replacement.lower()
        elif matched_text[0].isupper() and not matched_text[1:].isupper():
            if len(replacement) > 1:
                return replacement[0].upper() + replacement[1:].lower()
            else:
                return replacement.upper()
        elif matched_text[0].isupper() and matched_text[1:].isupper():
            return replacement.upper()
        else:
            result = []
            replacement_chars = list(replacement)
            replacement_index = 0

            for char in matched_text:
                if replacement_index >= len(replacement_chars):
                    break
                if char.isupper():
                    result.append(replacement_chars[replacement_index].upper())
                else:
                    result.append(replacement_chars[replacement_index].lower())
                replacement_index += 1

            if replacement_index < len(replacement_chars):
                result.extend(replacement_chars[replacement_index:])

            return ''.join(result)

    def replace_words_batch(self, text, rules, case_sensitive, match_whole_words, preserve_case):
        """
        Batch replacement using a single regex for all rules.
        Returns both the replaced text and a list of changes made.
        """
        if not rules:
            return text, []

        rules.sort(key=lambda x: len(x[0]), reverse=True)

        pattern_parts = []
        for old_phrase, new_phrase in rules:
            if match_whole_words:
                pattern = self.create_whole_word_pattern(old_phrase)
            else:
                pattern = self.escape_for_regex(old_phrase)
            pattern_parts.append(f'({pattern})')

        combined_pattern = '|'.join(pattern_parts)
        flags = 0 if case_sensitive else re.IGNORECASE

        changes = []
        result_parts = []
        last_end = 0

        def replace_func(match):
            nonlocal last_end

            result_parts.append(text[last_end:match.start()])

            for i, group in enumerate(match.groups()):
                if group is not None:
                    matched_text = group

                    for old_phrase, new_phrase in rules:
                        match_found = False

                        if not case_sensitive:
                            if matched_text.lower() == old_phrase.lower():
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase),
                                        matched_text, re.IGNORECASE):
                                match_found = True
                        else:
                            if matched_text == old_phrase:
                                match_found = True
                            elif re.match(self.create_whole_word_pattern(old_phrase) if match_whole_words else self.escape_for_regex(old_phrase),
                                        matched_text):
                                match_found = True

                        if match_found:
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched_text, new_phrase)
                            else:
                                replacement = new_phrase

                            changes.append({
                                'old': matched_text,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })

                            result_parts.append(replacement)
                            last_end = match.end()
                            return replacement

                    result_parts.append(matched_text)
                    last_end = match.end()
                    return matched_text

            result_parts.append(match.group(0))
            last_end = match.end()
            return match.group(0)

        try:
            re.sub(combined_pattern, replace_func, text, flags=flags)

            if last_end < len(text):
                result_parts.append(text[last_end:])

            result = ''.join(result_parts)

        except Exception as e:
            print(f"Regex error: {e}")
            result = text
            changes = []
            for old_phrase, new_phrase in rules:
                if match_whole_words:
                    pattern = self.create_whole_word_pattern(old_phrase)
                    rflags = 0 if case_sensitive else re.IGNORECASE

                    def fallback_replacer(match):
                        matched = match.group(0)
                        if preserve_case:
                            replacement = self.preserve_case_pattern(matched, new_phrase)
                        else:
                            replacement = new_phrase

                        changes.append({
                            'old': matched,
                            'new': replacement,
                            'start': match.start(),
                            'end': match.end(),
                            'rule_used': f"{old_phrase} → {new_phrase}"
                        })
                        return replacement

                    result = re.sub(pattern, fallback_replacer, result, flags=rflags)
                else:
                    if case_sensitive:
                        pos = 0
                        while True:
                            idx = result.find(old_phrase, pos)
                            if idx == -1:
                                break

                            changes.append({
                                'old': old_phrase,
                                'new': new_phrase,
                                'start': idx,
                                'end': idx + len(old_phrase),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })

                            result = result[:idx] + new_phrase + result[idx + len(old_phrase):]
                            pos = idx + len(new_phrase)
                    else:
                        pattern = re.escape(old_phrase)

                        def fallback_replacer_ci(match):
                            matched = match.group(0)
                            if preserve_case:
                                replacement = self.preserve_case_pattern(matched, new_phrase)
                            else:
                                replacement = new_phrase

                            changes.append({
                                'old': matched,
                                'new': replacement,
                                'start': match.start(),
                                'end': match.end(),
                                'rule_used': f"{old_phrase} → {new_phrase}"
                            })
                            return replacement

                        result = re.sub(pattern, fallback_replacer_ci, result, flags=re.IGNORECASE)

        return result, changes

    def format_changes_report(self, changes, format_type):
        """Create a human-readable report of all changes made."""
        if not changes:
            return "No changes were made."

        if format_type == "markdown":
            report = "### Changes Made:\n\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. **{change['old']}** → **{change['new']}**  \n"
            return report

        elif format_type == "html":
            report = "<h3>Changes Made:</h3>\n<ul>\n"
            for change in changes:
                report += f'  <li><b>{change["old"]}</b> → <b>{change["new"]}</b></li>\n'
            report += "</ul>"
            return report

        else:
            report = "Changes Made:\n"
            for i, change in enumerate(changes, 1):
                report += f"{i}. {change['old']} -> {change['new']}\n"
            return report

    def replace_words(self, text, replacement_rules, case_sensitive=False,
                       match_whole_words=True, sort_by_length=True, preserve_case=True,
                       highlight_format="markdown"):
        """Apply replacement rules to the input text and return both plain and highlighted versions."""
        if not text or not replacement_rules:
            return (text, text)

        rules = self.parse_rules(replacement_rules)

        if not rules:
            return (text, text)

        if sort_by_length:
            rules.sort(key=lambda x: len(x[0]), reverse=True)

        result, changes = self.replace_words_batch(
            text, rules, case_sensitive, match_whole_words, preserve_case
        )

        if changes:
            changes_sorted = sorted(changes, key=lambda x: x['start'], reverse=True)

            highlighted = text
            for change in changes_sorted:
                old_text = change['old']
                new_text = change['new']

                if highlight_format == "markdown":
                    highlighted = highlighted[:change['start']] + \
                                 f"**{new_text}**" + \
                                 highlighted[change['end']:]
                elif highlight_format == "html":
                    highlighted = highlighted[:change['start']] + \
                                 f'<span style="background-color: #ffff00; font-weight: bold;">{new_text}</span>' + \
                                 highlighted[change['end']:]
                else:
                    highlighted = highlighted[:change['start']] + \
                                 f"[{new_text}]" + \
                                 highlighted[change['end']:]

            changes_report = self.format_changes_report(changes, highlight_format)

            if highlight_format == "markdown":
                highlighted = f"{highlighted}\n\n{changes_report}"
            elif highlight_format == "html":
                highlighted = f"{highlighted}<br><br>{changes_report}"
            else:
                highlighted = f"{highlighted}\n\n{changes_report}"
        else:
            highlighted = text

        return (result, highlighted)

    def get_random_value(self, category_key, random_gen):
        """Get a random value from the options for a category, excluding 'none'."""
        options = self.CATEGORIES[category_key][0]
        # Filter out 'none' and any empty values
        valid_options = [opt for opt in options if opt and opt != "none"]
        if not valid_options:
            return "none"
        return random_gen.choice(valid_options)

    def build_rules_text(self, **kwargs):
        """Turn the active dropdown selections into a newline-delimited
        'old,new' rules string, using the built-in mapping tables for each
        category (these run entirely in the background, not as widgets)."""
        # Check if randomization is enabled
        randomize = kwargs.get('randomize', False)
        seed = kwargs.get('seed', 0)
        
        # Setup random generator if randomization is enabled
        random_gen = None
        if randomize:
            if seed == 0:
                # Use system randomness
                random_gen = random.Random()
            else:
                # Use deterministic seed
                random_gen = random.Random(seed)
        
        lines = []
        for cat_key in self.CATEGORIES:
            dropdown_val = (kwargs.get(cat_key) or "none").strip()
            
            # Skip special categories - they're handled separately
            if cat_key in self.SPECIAL_CATEGORIES:
                continue
            
            # If randomize is enabled and the category is set to "none", pick a random value
            if randomize and dropdown_val == "none" and cat_key in self.RANDOMIZABLE_CATEGORIES:
                dropdown_val = self.get_random_value(cat_key, random_gen)
            
            if not dropdown_val or dropdown_val == "none":
                continue

            actual_value = dropdown_val
            mapping_text = self.CATEGORIES[cat_key][1]

            for raw_line in mapping_text.strip().split("\n"):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) != 2:
                    continue

                old_phrase, template = parts
                if not old_phrase or not template:
                    continue

                new_phrase = template.replace("{value}", actual_value)

                # Skip no-op rules (source already equals the target text)
                if old_phrase.lower() == new_phrase.lower():
                    continue

                lines.append(f"{old_phrase},{new_phrase}")

        return "\n".join(lines)

    def replace_attributes(self, text, case_sensitive=False, match_whole_words=True,
                            sort_by_length=True, preserve_case=True,
                            highlight_format="markdown", randomize=False, seed=0, **kwargs):
        """
        Main entry point - applies all replacements including special categories.
        """
        if not text:
            return (text, text, "No text provided.")

        # First, handle special categories that need custom processing
        all_changes = []
        result_text = text

        # Process photograph type (fallback insertion)
        photograph_type_value = kwargs.get('photograph_type', 'none')
        if randomize and photograph_type_value == 'none':
            photograph_type_value = self.get_random_value('photograph_type', random.Random(seed) if seed else random.Random())
        
        if photograph_type_value != 'none':
            result_text, photo_changes = self.apply_photograph_type_fallback(
                result_text, photograph_type_value, preserve_case
            )
            all_changes.extend(photo_changes)

        # Build standard replacement rules
        rules_text = self.build_rules_text(
            randomize=randomize,
            seed=seed,
            **kwargs
        )

        # Apply standard replacements
        if rules_text:
            result_text, highlighted = self.replace_words(
                result_text, rules_text,
                case_sensitive=case_sensitive,
                match_whole_words=match_whole_words,
                sort_by_length=sort_by_length,
                preserve_case=preserve_case,
                highlight_format=highlight_format,
            )
        else:
            highlighted = result_text

        # If we had special category changes, update the highlighted text to include them
        if all_changes:
            # Rebuild the full changes list including standard replacements
            # For now, we'll just return the result without full highlighting for special changes
            # A more sophisticated approach would combine all changes
            pass

        # Create rules applied report
        rules_report = rules_text or "No standard rules applied."
        if all_changes:
            rules_report += f"\n\nSpecial category changes: {len(all_changes)}"

        return (result_text, highlighted, rules_report)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GRPromptReplacerAttributesBasic": GRPromptReplacerAttributesBasic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptReplacerAttributesBasic": "GR Prompt Replacer Attributes Basic"
}