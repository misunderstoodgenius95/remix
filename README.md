# House Remixer 🎛️

Software Python che trasforma una canzone in un **remix house**: rileva BPM, beat e
tonalità, porta il brano al tempo house, e costruisce un arrangiamento completo da DJ
(intro, build, drop, break, drop, outro) con batteria, basso e accordi sintetizzati che
seguono l'armonia originale.

Dipende solo da **numpy**. `ffmpeg` è opzionale e serve solo per leggere o scrivere
MP3/FLAC/M4A (i WAV funzionano senza).

## Installazione

```bash
pip install -r requirements.txt
```

## Uso rapido

```bash
python3 house_remixer.py canzone.mp3
```

Crea `canzone_house_classic.wav`.

```bash
python3 house_remixer.py canzone.mp3 -p deep --bpm 122 -o remix.mp3 --stems stems/
```

### Preset

| preset    | BPM | carattere                                                        |
|-----------|-----|------------------------------------------------------------------|
| `classic` | 124 | organ stab anni '90, basso in levare, open hat sui controtempi   |
| `deep`    | 120 | pad caldi, shaker con swing, basso morbido, più riverbero        |
| `tech`    | 126 | basso rolling a sedicesimi, ride, sidechain più aggressiva       |

### Opzioni principali

| opzione | descrizione |
|---|---|
| `-p, --preset` | `classic`, `deep`, `tech` |
| `--bpm N` | BPM del remix (default: quello del preset) |
| `--structure full\|short` | arrangiamento lungo (~80 battute) o radio edit (~64) |
| `--semitones N` | traspone il brano (es. `-2`, `+1.5`) |
| `--source-bpm N` | forza il BPM originale se il rilevamento sbaglia |
| `--shift-beats N` | sposta il downbeat di N beat se il remix risulta "sfasato" |
| `--no-lowcut` | non filtrare le basse del brano nei drop |
| `--source-gain / --drums-gain / --bass-gain` | livelli dei gruppi |
| `--stems DIR` | esporta anche `drums`, `bass`, `music`, `source`, `fx` in WAV, da importare in una DAW |
| `--seed N` | cambia le micro-variazioni (velocity, rumore) |

## Come funziona

1. **Beat tracking**: flusso spettrale, stima del tempo con un filtro a pettine
   (60–190 BPM), poi un tracking di **ogni singolo beat** con programmazione dinamica.
   Se il brano ha un tempo costante (musica elettronica, brani quantizzati) usa una
   griglia perfetta. Se il tempo varia (band dal vivo, brani suonati a mano) segue la
   deriva beat per beat. I beat senza un attacco chiaro (intro senza batteria, pad)
   vengono ricavati dai vicini affidabili.
2. **Time-stretch con warping**: phase vocoder stereo-coerente con mappa temporale
   variabile. Ogni beat del brano finisce esattamente sulla griglia del remix e la
   tonalità non cambia. Il reset di fase sui transienti mantiene gli attacchi netti e
   a tempo (errore tipico sotto i 2 ms).
3. **Armonia**: chromagram con profili di Krumhansl per la tonalità, e analisi della
   banda bassa battuta per battuta. Il nuovo basso e gli accordi seguono così le note
   della canzone.
4. **Arrangiamento**: la frase da 8 battute più energica diventa il "drop" e quella più
   calma il "break". Si aggiungono filtri passa-basso/passa-alto in sweep, riser, rullate
   di rullante, impact, sidechain sulla cassa, eco ping-pong e riverbero a convoluzione.
5. **Master**: soft-clip e limitazione del picco a −1 dBFS.

## Provarlo senza una canzone

```bash
python3 make_demo.py -o demo.wav --bpm 100
```

```bash
python3 house_remixer.py demo.wav -p classic
```

`make_demo.py` genera un brano pop originale da usare come input di prova.

> Nota: usa il software su brani di cui hai i diritti, o per uso personale.
> La pubblicazione di remix di brani altrui richiede l'autorizzazione dei titolari.
