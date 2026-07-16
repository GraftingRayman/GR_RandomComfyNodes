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
    """

    # ---- default mapping tables -------------------------------------------------
    # Each line: source_phrase,template   (template uses {value} as a placeholder)

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

    # ---- CLOTHING ATTRIBUTES -------------------------------------------------
    
    # Top Colors
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

    # Top Types
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

    # Bottom Colors
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

    # Bottom Types
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

    # Footwear Colors
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

    # Footwear Types
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

    # Outfit Style (combines top and bottom for dresses/suits etc.)
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

    # category_key -> (dropdown options, default mapping text)
    CATEGORIES = {
        "hair_color": (HAIR_COLOR_OPTIONS, HAIR_COLOR_MAPPING_DEFAULT),
        "skin_tone": (SKIN_TONE_OPTIONS, SKIN_TONE_MAPPING_DEFAULT),
        "hair_style": (HAIR_STYLE_OPTIONS, HAIR_STYLE_MAPPING_DEFAULT),
        "eye_color": (EYE_COLOR_OPTIONS, EYE_COLOR_MAPPING_DEFAULT),
        "smile": (SMILE_OPTIONS, SMILE_MAPPING_DEFAULT),
        "expression": (EXPRESSION_OPTIONS, EXPRESSION_MAPPING_DEFAULT),
        "pose": (POSE_OPTIONS, POSE_MAPPING_DEFAULT),
        "hand_position": (HAND_POSITION_OPTIONS, HAND_POSITION_MAPPING_DEFAULT),
        "head_position": (HEAD_POSITION_OPTIONS, HEAD_POSITION_MAPPING_DEFAULT),
        "top_color": (TOP_COLOR_OPTIONS, TOP_COLOR_MAPPING_DEFAULT),
        "top_type": (TOP_TYPE_OPTIONS, TOP_TYPE_MAPPING_DEFAULT),
        "bottom_color": (BOTTOM_COLOR_OPTIONS, BOTTOM_COLOR_MAPPING_DEFAULT),
        "bottom_type": (BOTTOM_TYPE_OPTIONS, BOTTOM_TYPE_MAPPING_DEFAULT),
        "footwear_color": (FOOTWEAR_COLOR_OPTIONS, FOOTWEAR_COLOR_MAPPING_DEFAULT),
        "footwear_type": (FOOTWEAR_TYPE_OPTIONS, FOOTWEAR_TYPE_MAPPING_DEFAULT),
        "outfit_style": (OUTFIT_STYLE_OPTIONS, OUTFIT_STYLE_MAPPING_DEFAULT),
    }

    # List of attribute categories for randomization
    RANDOMIZABLE_CATEGORIES = [
        "hair_color", "skin_tone", "hair_style", "eye_color", "smile", 
        "expression", "pose", "hand_position", "head_position",
        "top_color", "top_type", "bottom_color", "bottom_type",
        "footwear_color", "footwear_type", "outfit_style"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for cat_key, (options, mapping_default) in cls.CATEGORIES.items():
            label = cat_key.replace("_", " ").title()
            optional[cat_key] = (options, {"default": "none", "label": label})

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

    # ---- self-contained replacement engine (no dependency on GRPromptReplacer.py) --

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
        rules_text = self.build_rules_text(
            randomize=randomize,
            seed=seed,
            **kwargs
        )

        if not text or not rules_text:
            return (text, text, rules_text or "No categories selected.")

        result, highlighted = self.replace_words(
            text, rules_text,
            case_sensitive=case_sensitive,
            match_whole_words=match_whole_words,
            sort_by_length=sort_by_length,
            preserve_case=preserve_case,
            highlight_format=highlight_format,
        )

        return (result, highlighted, rules_text)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GRPromptReplacerAttributesBasic": GRPromptReplacerAttributesBasic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GRPromptReplacerAttributesBasic": "GR Prompt Replacer Attributes Basic"
}