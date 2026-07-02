# Roadmap — stan względem specyfikacji

Legenda: ✅ zrobione | 🟡 częściowo | ⬜ planowane

## Silnik

| Obszar | Stan | Notatki |
|---|---|---|
| Okno + kontekst GL 3.3 (GLFW/ModernGL) | ✅ | raw mouse motion, vsync, fullscreen |
| Architektura warstwowa engine/game | ✅ | silnik nie zna gameplayu |
| Bloki data-driven (JSON) | ✅ | `configs/blocks.json` → tablice NumPy |
| Chunki + streaming async | ✅ | pula wątków, priorytet wg odległości i kierunku kamery, budżety per klatka |
| Mesher (culling + AO + kompresja wierzchołków) | ✅ | zwektoryzowany NumPy, 8 B/wierzchołek |
| Greedy meshing | ⬜ | konflikt z per-vertex AO — wymaga benchmarku (spec: najpierw mierz) |
| Frustum culling | ✅ | zwektoryzowany test AABB vs 6 płaszczyzn |
| Occlusion culling / LOD | ⬜ | |
| Tekstury proceduralne (texture array + mipmapy) | ✅ | 25 kafelków, zero assetów |
| Oświetlenie kierunkowe + AO + emisja (lawa) | ✅ | per-vertex; bez propagacji światła blokowego |
| Światło blokowe (flood fill), kolorowe | ⬜ | kolejny duży krok |
| Cykl dnia/nocy, niebo proceduralne, gwiazdy, mgła | ✅ | |
| Zapis świata (tylko zmodyfikowane chunki, npz + meta) | ✅ | odporny na uszkodzone pliki (regeneracja) |
| Profiler wbudowany | 🟡 | timingi w smoke_test + F3; brak per-stage profilera |
| Audio (OpenAL) | ⬜ | |
| ECS | ⬜ | na razie jedyną encją jest gracz; ECS wejdzie z mobami |
| Multiplayer (serwer autorytatywny) | ⬜ | architektura world/streaming już rozdziela dane od renderu |
| Modding / skrypty | 🟡 | bloki/kafelki data-driven; brak ładowania paczek modów |

## Świat

| Obszar | Stan | Notatki |
|---|---|---|
| Wysokości: kontynenty + wzgórza + góry ridged + domain warp | ✅ | |
| Klimat: temperatura/wilgotność → biomy emergentne | ✅ | ocean, plaża, pustynia, śnieg, góry, las, równiny |
| Jaskinie spaghetti + caverny + lawa | ✅ | nie przebijają powierzchni (brak symulacji cieczy) |
| Rudy (węgiel/żelazo/złoto/diament wg głębokości) | ✅ | |
| Drzewa bezszwowe przez granice chunków | ✅ | stateless hash świata |
| Rośliny (trawa, kwiaty) | ✅ | |
| Rzeki, jeziora, wioski, struktury | ⬜ | pipeline przebiegów jest na to gotowy |
| Symulacja cieczy (rozlewanie) | ⬜ | woda statyczna na poziomie morza |
| Pogoda (deszcz/śnieg/burza) | ⬜ | |

## Gameplay

| Obszar | Stan | Notatki |
|---|---|---|
| Ruch: chodzenie, sprint, skok, pływanie, latanie | ✅ | kolizje AABB per oś |
| Kopanie/stawianie/pobieranie bloku, podświetlenie celu | ✅ | raycast DDA, ochrona przed stawianiem w sobie |
| Hotbar + HUD + F3 + pauza | ✅ | |
| Ekwipunek, crafting, przetrwanie (HP/głód), moby | ⬜ | kolejne etapy |

## Znane kompromisy wydajnościowe (do zmierzenia przed optymalizacją)

- Generacja ~60 ms/chunk — dominują szumy 3D jaskiń/rud (8× perlin3);
  kandydaci: pół-rozdzielczość + interpolacja, Numba, mniejsza liczba pól.
- Zapis zmodyfikowanego chunka przy unload odbywa się na głównym wątku
  (kilka ms) — przenieść do puli wątków.
- ~450 draw calli przy render distance 10 — wystarcza (60 FPS na RTX 3050);
  przy większych dystansach rozważyć merged/indirect drawing.
