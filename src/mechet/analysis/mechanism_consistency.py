"""A reaction precedent does not verify a symbolic mechanism or intermediate."""


def assess_mechanism(reaction, evidence):
    accepted = reaction.get('formal_status') == 'recorded_executor_accepted_complete_reaction'
    return {'occurrence_id':reaction['occurrence_id'],'reaction_key':reaction.get('reaction_key'),
            'run_id':reaction['run_id'],'target_id':reaction['target_id'],'source':reaction['source'],
            'mechanistic_consistency':'formal_only' if accepted else 'unresolved',
            'electron_program_source_pointer':reaction.get('electron_program_source_pointer'),
            'net_reaction_evidence_tier':evidence.get('tier','C'),
            'physical_intermediates_experimentally_verified':False,
            'basis':'original executor record only; net experimental precedent is not an arrow-order observation'}
