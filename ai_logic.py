import config

def ai_brain_decision(intersections_data):
    actions = []
    dir_map = {'N': 0, 'S': 1, 'E': 2, 'O': 3}
    directions = ['N', 'S', 'E', 'O']

    # Mappa di inversione: Se vedo Nord, attivo Sud.
    swap_map = {'N': 'S', 'S': 'N', 'E': 'O', 'O': 'E'}

    for i, data in enumerate(intersections_data):
        queues = data['queues']
        emerg = data['emergencies']
        current_idx = data['current_phase']
        decision = -1

        # 1. EMERGENZA
        for d in directions:
            if emerg[d]:
                # Se INVERT_LOGIC è True, scambiamo la direzione target
                target_d = swap_map[d] if config.INVERT_DIRECTION_LOGIC else d
                decision = dir_map[target_d]
                break
        if decision != -1:
            actions.append(decision)
            continue

        # 2. CODE
        current_dir_char = directions[current_idx]

        # Se siamo invertiti, la fase corrente 'S' sta servendo la coda 'N'
        # Quindi per vedere se la fase attuale è utile, dobbiamo guardare la coda opposta
        served_queue_label = swap_map[current_dir_char] if config.INVERT_DIRECTION_LOGIC else current_dir_char
        current_queue_len = queues[served_queue_label]

        max_q = 0
        best_dir_label = served_queue_label

        # Troviamo quale direzione (Sensore) ha più coda
        for d in directions:
            if queues[d] > max_q:
                max_q = queues[d]
                best_dir_label = d  # Es. 'N' ha più coda

        # Ora decidiamo quale fase attivare per servire 'best_dir_label'
        target_phase_label = swap_map[best_dir_label] if config.INVERT_DIRECTION_LOGIC else best_dir_label
        target_phase_idx = dir_map[target_phase_label]

        if target_phase_idx != current_idx:
            if max_q > current_queue_len + 1:
                decision = target_phase_idx
            else:
                decision = current_idx
        else:
            decision = current_idx

        if max_q == 0: decision = current_idx
        actions.append(decision)

    return actions
