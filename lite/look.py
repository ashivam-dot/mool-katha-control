"""How the channel's AI paintings look: one style for every picture, and one fixed appearance per character, so
Rama is the same Rama in every beat and every Short. The writer describes each beat's scene in English; this
module turns that into the image model's prompt."""

from __future__ import annotations

import re
import zlib

STYLE = ("Devotional Indian epic oil painting in the tradition of Raja Ravi Varma with rich Pahari miniature colour:"
         " saffron, vermilion, deep indigo and burnished gold, warm cinematic rim light, soft atmospheric depth,"
         " painterly brushwork, dignified and reverent mood, vertical 9:16 composition with the subject in the"
         " upper two thirds")
CLOSE = "Intimate close-up portrait, the face and its emotion fill the frame, eyes sharp and expressive."
# A close-up may be of hands or an object, so it doesn't ask for a face.
DETAIL = "Intimate close-up, the subject fills the frame in sharp detail, shallow depth of field."
SAFE = ("Fully clothed figures, no blood, no gore, no nudity. No text, no letters, no words, no watermark,"
        " no signature, no frame, no border.")

# English names the writer uses in scene descriptions, and how each one always looks.
CHARACTERS: tuple[tuple[str, str], ...] = (
    (r"\bRama\b|\bRam\b|Raghava", "Lord Rama: serene young man in his twenties (an adult, never a child, as tall as"
     " the men beside him) with blue skin, gentle lotus eyes, golden crown, long black hair, yellow silk dhoti and"
     " upper cloth, a tall bow and a quiver of arrows"),
    (r"Lakshmana|Lakshman", "Lakshmana: young man in his twenties (an adult, never a child) with fair golden skin,"
     " golden crown, saffron dhoti, bow, alert and devoted"),
    (r"\bSita\b|Seetha|Janaki|Vaidehi", "Sita: graceful dignified princess, red and gold silk saree covering her"
     " fully, gold jewellery, long black braid"),
    (r"Hanuman|Hanuma\b|Maruti", "Hanuman: mighty vanara monkey god with a distinctly monkey face and muzzle, wise"
     " kind eyes, golden-orange fur on his body, red-gold crown, gold earrings, a golden mace (gada), long tail"),
    (r"Ravana|Dashanana", "Ravana: ten-headed rakshasa king, dark skin, ten golden crowns, ornate golden armour,"
     " thick moustaches, proud and imposing"),
    (r"Dasharatha|Dashratha", "King Dasharatha: aged king with a white beard, heavy golden crown, crimson and gold"
     " royal robes"),
    (r"Kaikeyi", "Queen Kaikeyi: proud queen in deep purple silk and gold jewellery"),
    (r"Kausalya", "Queen Kausalya: gentle elderly queen in white and gold silk"),
    (r"Bharata", "Bharata: young prince with dark skin in simple ascetic clothes, matted hair, sorrowful"),
    (r"Shatrughna", "Shatrughna: young prince with fair skin, golden crown, blue dhoti"),
    (r"Vishvamitra|Vishwamitra", "sage Vishvamitra: tall ascetic with a long white beard, matted hair tied high,"
     " saffron robes, rudraksha beads, piercing fiery eyes"),
    (r"Vasishtha|Vashishtha", "sage Vasishtha: calm old sage with a flowing white beard, white robes, holding a"
     " wooden staff"),
    (r"Valmiki", "sage Valmiki: old sage with a white beard in white robes beside a river"),
    (r"Jatayu", "Jatayu: a real giant old vulture bird (not a man, no clothes, no crown) with grey-brown feathers,"
     " a bare wrinkled neck and a white ruff"),
    (r"Sampati", "Sampati: a real giant aged vulture bird (not a man) with burnt, featherless wings"),
    (r"Sugriva", "Sugriva: vanara king with golden-brown fur, a gold crown and necklace"),
    (r"\bVali\b|\bBali\b", "Vali: powerful vanara king with bright golden fur and a golden necklace"),
    (r"Angada", "Angada: young vanara prince with golden fur and a gold armband"),
    (r"Jambavan|Jambavat", "Jambavan: wise old bear king with grey fur and a crown"),
    (r"Vibhishana", "Vibhishana: calm rakshasa prince with dark skin, white and gold robes, folded hands"),
    (r"Kumbhakarna", "Kumbhakarna: colossal rakshasa giant with dark skin, heavy-lidded eyes, golden armour"),
    (r"Indrajit|Meghanada", "Indrajit: young rakshasa prince in dark gold armour with a bow, among storm clouds"),
    (r"Surpanakha|Shurpanakha", "Surpanakha: rakshasi princess in dark red robes, fierce"),
    (r"Maricha", "Maricha: rakshasa with dark skin who becomes a shining golden deer with silver spots"),
    (r"Kabandha", "Kabandha: huge headless rakshasa with a face in his belly and long arms"),
    (r"Shabari", "Shabari: very old devoted woman ascetic in simple saffron cloth, holding a basket of berries"),
    (r"Ahalya", "Ahalya: serene woman ascetic in white, glowing with light"),
    (r"Parashurama", "Parashurama: fierce sage-warrior with matted hair, white beard, saffron robes and a great axe"),
    (r"Bhagiratha", "Bhagiratha: ascetic king with matted hair in simple robes, arms raised in prayer"),
    (r"\bGanga\b", "Goddess Ganga: radiant goddess in white and silver flowing as a luminous river from the sky"),
    (r"\bShiva\b|Mahadeva|Shankara", "Lord Shiva: serene, blue-grey skin, matted hair holding a crescent moon,"
     " third eye, a serpent around his neck, tiger-skin garment, trident"),
    (r"\bVishnu\b|Narayana", "Lord Vishnu: four-armed god with blue skin holding a conch, discus, mace and lotus,"
     " golden crown"),
    (r"\bBrahma\b", "Lord Brahma: four-faced elderly god with white beards seated on a lotus"),
    (r"\bIndra\b", "Indra: king of the gods with a golden crown and thunderbolt, on a white elephant"),
    (r"\bAgni\b", "Agni: god of fire with flame-coloured skin rising from fire"),
    (r"\bVayu\b", "Vayu: god of wind with flowing hair and windswept robes"),
    (r"Garuda", "Garuda: golden eagle-man with vast wings"),
    (r"Kapila", "sage Kapila: ascetic with a white beard meditating, glowing"),
    (r"Krishna", "Lord Krishna: blue skin, peacock-feather crown, yellow silk dhoti, flute, gentle smile"),
    (r"Arjuna", "Arjuna: warrior prince with a golden crown, armour and the great bow Gandiva"),
    (r"Yudhishthira", "Yudhishthira: calm eldest Pandava king with a crown and white robes"),
    (r"Bhima\b", "Bhima: huge powerful Pandava warrior with a mace"),
    (r"Draupadi", "Draupadi: dignified queen with dark skin in a deep red saree covering her fully"),
    (r"Bhishma", "Bhishma: aged warrior patriarch with a long white beard, white robes and armour"),
    (r"Karna", "Karna: warrior with golden armour and golden earrings fused to his body"),
    (r"Drona", "Drona: old Brahmin warrior teacher with a white beard and a bow"),
    (r"Kunti", "Kunti: dignified queen mother in white silk"),
    (r"Gandhari", "Gandhari: queen with a cloth tied over her eyes"),
    (r"Savitri", "Savitri: devoted princess in a red saree"),
    (r"Yama\b|Yamaraja", "Yama: god of death with dark skin on a black buffalo, holding a noose, solemn not scary"),
)
MAX_CHARACTERS = 3


# When both characters are in a scene the model must be told the mood outright; on its own it paints them as a couple.
TOGETHER: tuple[tuple[str, str, str], ...] = (
    (r"\bSita\b|Seetha|Janaki|Vaidehi", r"Ravana|Dashanana",
     "Sita is Ravana's captive: her face is full of terror and tears, she turns away and struggles to break free,"
     " standing apart from him."),
)


def characters(scene: str) -> list[str]:
    """Fixed descriptions of the characters a scene names, in the order they appear in it."""
    found = []
    for pattern, look in CHARACTERS:
        hit = re.search(pattern, scene)
        if hit:
            found.append((hit.start(), look))
    return [look for _, look in sorted(found)][:MAX_CHARACTERS]


def prompt(scene: str, *, hook: bool = False, close: bool = False) -> str:
    parts = [scene.strip().rstrip(".") + "."]
    if hook:
        parts.append(CLOSE)
    elif close:
        parts.append(DETAIL)
    looks = characters(scene)
    if looks:
        parts.append("Characters: " + "; ".join(looks) + ".")
    parts += [mood for a, b, mood in TOGETHER if re.search(a, scene) and re.search(b, scene)]
    parts += [STYLE + ".", SAFE]
    return " ".join(parts)


def seed(episode: str, beat: int) -> int:
    return zlib.crc32(f"{episode}:{beat}".encode()) % 2_000_000_000


def plan(episode: str, shots: list[dict]) -> list[dict]:
    """One image prompt and seed per shot (shots.plan); a shot without a scene gets no picture of its own. The
    paint jobs and their files still call a picture's number "beat"."""
    out = []
    for i, shot in enumerate(shots):
        scene = (shot.get("scene") or "").strip()
        text = prompt(scene, hook=i == 0, close=bool(shot.get("close"))) if scene else ""
        out.append({"beat": i, "prompt": text, "seed": seed(episode, i)})
    return out
