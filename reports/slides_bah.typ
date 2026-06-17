#import "@preview/touying:0.7.4": *
#import themes.simple: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#show: show-theorion

#let accent = rgb("#9e2c2c")
#let accent-light = rgb("#f7e8e8")
#let code-bg = rgb("#f0d6d6")

#let info-box(body, size: 0.8em) = block(
  fill: accent-light,
  stroke: (left: 3pt + accent),
  radius: 4pt,
  inset: 10pt,
  width: 100%,
  text(size: size, body),
)


#show raw: set block(fill: code-bg, radius: 4pt, inset: (x: 8pt, y: 6pt), stroke: (top: 2pt + accent))


#show: simple-theme.with(
  aspect-ratio: "16-9",
  // align: horizon,
  // config-common(handout: true),
  config-common(frozen-counters: (theorem-counter,)), // freeze theorem counter for animation
  config-info(
    title: [Dataset BAH],
    author: [Matheus Girardi],
    institution: [Pucpr],
  ),
)

#set text(font: "Atkinson Hyperlegible", size: 0.9em)
#show raw.where(block: true): set text(font: "CommitMono Nerd Font")

#show heading.where(level: 1): it => {
  it
  line(stroke: 3pt + accent, length: 100%)
}

#show heading.where(level: 3): it => block(
  stroke: (left: 4pt + accent),
  inset: (left: 8pt, top: 4pt, bottom: 4pt, right: 0pt),
  it
)

#title-slide[
  = Dataset BAH
  #v(2em)


]

=== Overview do dataset

O dataset conta com 300 participantes, cada participante foi gravado respondendo a 7 perguntas distintas. Cada vídeo é a resposta do participante a uma das perguntas.
O foco do dataset é a detecção de ambivalência/hesitação nas respostas, as anotações foram feitas por especialistas a nível de frames e a tarefa final é a nível de vídeo.

= Estrutura do dataset
=== Pasta `Videos`

Contém os vídeos brutos por participante e por perguntas:

```
Videos/
└── 82557/
    └── Visite_1/
        ├── 82557_Question_3_2024-08-22_14-48-26 Video.mp4
        ├── 82557 Question 5 2024-08-22 14-48-26 Video.mp4
        └── 82557 Question 6 2024-08-22 14-48-26 Video.mp4
```
---
=== Pasta  `cropped-aligned-faces`:

Contém os rostos dos participantes pré-cortados e alinhados, para cada frame:

```
Videos/
└── 82555/
    └── Visite_1/
        └── 82555_Question_3_2024-08-22_12-25-53_Video.mp4/
            ├── frame-1.jpg
            ├── frame-2.jpg
            ├── frame-3.jpg
            └── frame-*.jpg
```

=== Pasta `transcription`

Contém a transcrição de cada vídeo, tanto completa como por trechos com timestamps e detecção da língua. A transcrição foi feita com o modelo Whisper. 

#align(center)[ #text(size: 0.65em)[
```yaml
chunks:
- language: english
  text: ' I get a lot of joy doing things around the house, some DIY projects and
    overall just producing something that I can physically touch.'
  timestamp: !!python/tuple
  - 0.0
  - 12.0
- language: english
  text: ' In my job I make a lot of decisions but not a lot of concrete results.'
  timestamp: !!python/tuple
  - 12.0
  - 16.88
text: ' I get a lot of joy doing things around the house, some DIY projects and overall
  just producing something that I can physically touch. In my job I make a lot of
  decisions but not a lot of concrete results.'
```
]]

=== Pastas `split`, `split-frames`
*COLOCAR AS PORCENTAGENS DOS SPLITS*


Ambas as pastas possuem arquivos `.txt` que separam os participantes em treino, teste e validação. 

Na pasta `split` cada linha dos arquivos `.txt` segue:

#info-box(size: 1em)[
`video_id, classe_do_video,transcrição_do_video_completa`
]
Já na pasta `split-frames` cada linha segue:

#info-box(size: 1em)[
`frame_id, classe_do_frame`
]

#align(horizon)[
Em ambos os casos a classe é binária (1 para ambivalência, 0 para nada) e o identificador é o caminho para o vídeo (com o frame ao final para `split-frames`)
#info-box()[
`Videos/82813/Visite_1/82813_Question_3_2025-01-31_19-18-58_Video.mp4/frame-0.jpg`
  ]
]

---

=== Arquivos

- *`extract_frames_from_videos.sh`* e *`extract_frames_from_videos.py`*: servem para realizar a extração dos frames brutos de todos os vídeos, populando a pasta `Frames`.

- *`bah-video.csv`*: arquivo com todos os ids dos vídeos e sua respectiva classe.

- *`BAH_Dataset_EULA-2.pdf`*: End User Licence Agreement - EULA do dataset.

- *`video_annotation_transcript.yaml`*: possui informações detalhadas sobre as anotações do dataset, o arquivo é um dicionário python cuja chave é o id do vídeo e as seguintes informações:

#info-box[
  - _all_cues_: pistas usadas (áudio, corpo, rosto, linguagem e inconsistências).
  - _annotator_id_: ID do anotador.
  - _certainty_ah_: confiança do anotador nas marcações.
  - _fr_detailed_ah_: intervalo em frames com A/H.
  - _frame_annotation_: rótulo por frame (1 = tem A/H, 0 = não tem).
  - _global_ah_: rótulo geral do vídeo (1/0).
  - _time_detailed_ah_: intervalo em tempo com A/H.
  - _transcript_: fala segmentada + transcrição completa (text).
]

#info-box[
Segue um exemplo do arquivo no próximo slide.
]
---
#align(center)[ #text(size: 0.61em)[
```yaml
Videos/82553/Visite_1/82553_Question_1_2024-08-22_12-11-55_Video.mp4:
  all_cues:
  - audio: null
    body: null
    facial: null
    inconsistencies: null
    language: null
  - audio: null
    body: null
    facial: null
    inconsistencies: null
    language: null
  annotator_id: MGG
  certainty_ah:
  - 1
  fr_detailed_ah:
  - - 51
    - 95
  frame_annotation:
  - - Videos/82553/Visite_1/82553_Question_1_2024-08-22_12-11-55_Video.mp4/frame-0.jpg
    - 0
  - - Videos/82553/Visite_1/82553_Question_1_2024-08-22_12-11-55_Video.mp4/frame-1.jpg
    - 0
```
]]
---
=== Meta data

- *`meta_data.yml`* consiste em um dicionário com 1 entrada por participante com os seguintes dados:

#info-box[
  - idade e faixa etária
  - país de nascimento
  - província do Canadá onde mora
  - etnia e etnia simplificada
  - gênero
  - se é estudante
  - permissão de uso em publicações
  - permissão de uso em desafios (challenges)
]

- Exemplo arquivo de metadata:
#align(center)[ #text(size: 0.65em)[
```yaml
'82553':
  Age: 51
  Age range: 45 - 54 years
  Birth country: Canada
  Canada province: Ontario (ON)
  Ethnicity: East Asian (Chinese, Korean, Japanese, and/or Taiwanese descent)
  Ethnicity simplified: Asian
  Gender: Male
  Is student: 'No'
  Use in challenges: 'Yes'
  Use in publications: 'Yes'
```
]]
---

Para carregar os arquvios .yml do dataset os autores recomendam instalar PyYAML (`pip install pyyaml`) e usar:

#align(center)[ #text(size: 0.8em)[
```py
import yaml

with open('meta_data.yml', 'r') as f:
  content = yaml.safe load(f)
   # ou: 
  content = yaml.full load(f)
```
]
]