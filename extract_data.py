#!/usr/bin/env python3
import re
import csv
from datetime import datetime

def extract_fields_from_line(line):
    """
    Estrae i campi da una singola riga di log usando un approccio step-by-step.
    Ritorna None se la riga non contiene tutti i campi richiesti.
    """
    # 1. Trova timestamp dividendo i gruppi (ore, minuti, secondi, millisecondi)
    time_match = re.search(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]', line)
    if not time_match:
        return None

    # 2. Trova mvmt e thr (dopo il timestamp)
    rest_after_time = line[time_match.end():]
    mvmt_match = re.search(r'mvmt:([\d.]+)\s+thr:([\d.]+)', rest_after_time)
    if not mvmt_match:
        return None

    # 3. Trova stato (dopo mvmt)
    rest_after_mvmt = rest_after_time[mvmt_match.end():]
    state_match = re.search(r'\s*\|\s*(\w+)\s*\|', rest_after_mvmt)
    if not state_match:
        return None

    # 4. Trova pacchetti
    rest_after_state = rest_after_mvmt[state_match.end():]
    pks_match = re.search(r'(\d+)\s+pkt/s', rest_after_state)
    if not pks_match:
        return None

    # 5. Trova canale e RSSI
    rest_after_pks = rest_after_state[pks_match.end():]
    ch_match = re.search(r'ch:(\d+)\s+rssi:(-?\d+)', rest_after_pks)
    if not ch_match:
        return None


    mvmt = float(mvmt_match.group(1))
    thr = float(mvmt_match.group(2))
    return {
            'ore': int(time_match.group(1)),
            'minuti': int(time_match.group(2)),
            'secondi': int(time_match.group(3)),
            'millisecondi': int(time_match.group(4)),
            'mvmt': mvmt,
            'thr': thr,
            'motion_index': round(mvmt / thr, 6) if thr != 0 else None,#arrotondo il Motion Index a un numero fisso di cifre decimali: 6
            'picco': 1 if mvmt > thr else 0,  # se mvmt>thr allore 1 se mvmt<thr allora 0
            'state': state_match.group(1),
            'pkt_s': int(pks_match.group(1)),
            'channel': int(ch_match.group(1)),
            'rssi': int(ch_match.group(2))
        }
    

def extract_data(input_file, output_file):
    """
    Estrae dati da un file di log e li salva in un CSV strutturato.

    Pattern delle righe:
    [18:20:04.407][I][espectre:412][wifi]: [#--------------|----] 8% | mvmt:0.1958 thr:2.2238 | MOTION | 57 pkt/s | ch:11 rssi:-46
    """

    with open(input_file, 'r') as f_in:
        content = f_in.read()
        
        # RILEVAZIONE AUTOMATICA:
        # Se nel testo è presente la sequenza ';;', divide usando quella
        if ';;' in content:
            lines = content.split(';;')
        else:
            # Altrimenti divide usando i normali a capo del sistema (\n o \r\n)
            lines = content.splitlines()

    # Estrae i campi da ogni riga
    rows = []
    for line in lines:
        result = extract_fields_from_line(line)
        if result:
            rows.append([
                result['ore'],
                result['minuti'],
                result['secondi'],
                result['millisecondi'],
                result['mvmt'],
                result['thr'],
                result['motion_index'],
                result['picco'],
                result['state'],
                result['pkt_s'],
                result['channel'],
                result['rssi']
            ])

    # Scrive il CSV
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # Scrive l'header
        writer.writerow([
            'ora',
            'minuti',
            'secondi',
            'millisecondi',
            'mvmt',
            'thr',
            'motion_index',
            'picco',
            'state',
            'pkt_s',
            'channel',
            'rssi'
        ])
        # Scrive le righe
        writer.writerows(rows)

    print(f'Estratti {len(rows)} record.')
    print(f'File di output salvato in: {output_file}')

    # Mostra le prime righe come preview
    if rows:
        print('\nPreview delle prime 5 righe:')
        for i, row in enumerate(rows[:5]):
            print(f'{i+1}. {row}')
    else:
        print('\nNessun dato estratto. Controlla che il file contenga righe nel formato atteso.')

if __name__ == '__main__':
    input_file = 'test.csv'
    output_file = 'dati_estratti_test.csv'

    extract_data(input_file, output_file)
