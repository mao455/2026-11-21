"""Initial overhead cue comes from the live renderer."""
def paired_initial_rgb(case,rgb,metadata,public):
    if case.get('initial_overhead_reference'):raise ValueError('Archived input is not part of the minimal live entry')
    return rgb
