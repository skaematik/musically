from music21 import converter


bass = "CDEFGAB"
treble = bass.lower()

bass = list(bass)
treble = list(treble)

higher_treble = [a+"'" for a in treble]
even_higher_treble = [a+"'" for a in higher_treble]

# print(bass)
# print(treble)
# print(higher_treble)
# print(even_higher_treble)

fullrange_up = bass + treble + higher_treble + even_higher_treble
fullrange_down = fullrange_up[::-1]
fullrange = fullrange_up + fullrange_down

all_rests = converter.parse("tinynotation: 4/4 r1 r2 r4 r8 r16")
# all_rests.show()

all_crotchets = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"4" for a in fullrange]))
all_halfs = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"2" for a in fullrange]))
all_wholes = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"1" for a in fullrange]))

single_quaver_range = [a+"8 r8" for a in fullrange]
single_semiquaver_range = [a+"16 r16" for a in fullrange]

all_single_quavers = converter.parse(
    "tinynotation: 4/4 " + " ".join(single_quaver_range))
all_single_semiquavers = converter.parse(
    "tinynotation: 4/4 " + " ".join(single_semiquaver_range))

all_multi_quavers = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"8" for a in fullrange]))
all_multi_semiquavers = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"16" for a in fullrange]))

# all_halfs.show()
# all_wholes.show()
# all_single_quavers.show()
# all_single_semiquavers.show()
# all_multi_quavers.show()
# all_multi_semiquavers.show()

fullrange_sharps = [a+"#" for a in fullrange]
fullrange_flats = [a+"-" for a in fullrange]
fullrange_acci = fullrange_sharps + fullrange_flats

all_crotchets_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"4" for a in fullrange_acci]))
all_halfs_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"2" for a in fullrange_acci]))
all_wholes_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"1" for a in fullrange_acci]))

single_quaver_range_acci = [a+"8 r8" for a in fullrange_acci]
single_semiquaver_range_acci = [a+"16 r16" for a in fullrange_acci]

all_single_quavers_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join(single_quaver_range_acci))
all_single_semiquavers_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join(single_semiquaver_range_acci))

all_multi_quavers_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"8" for a in fullrange_acci]))
all_multi_semiquavers_acci = converter.parse(
    "tinynotation: 4/4 " + " ".join([a+"16" for a in fullrange_acci]))

# all_crotchets_acci.show()
# all_halfs_acci.show()
# all_wholes_acci.show()
# all_single_quavers_acci.show()
# all_single_semiquavers_acci.show()
# all_multi_quavers_acci.show()
# all_multi_semiquavers_acci.show()

