"""
Modulo comun de carga de costes para el TSP cerrado de las 47 capitales.

Punto unico de acceso a las matrices dirigidas que necesitan tanto el ACO
como el GA:

    C(valor_hora) = valor_hora * tiempo_h + tax_eur + gas_eur

Los datos de entrada proceden de:

  datos_rutas/rutas_costes.csv   distancias, tiempos, peajes y combustible
  datos_rutas/ciudades.csv       latitud/longitud de las 47 capitales

Fuentes documentadas de esos datos en datos_rutas/fuentes.csv y
datos_rutas/tabla_peajes_2026.csv (consultadas el 2026-10-09).

Ejemplo de uso (tanto para el ACO como para el GA):

    from costes_comunes import cargar_costes
    datos = cargar_costes()
    M = datos.matriz_coste(valor_hora=20.0)   # matriz 47x47 a minimizar
    datos.coste_tour([...], valor_hora=20.0)  # coste de un ciclo cerrado
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos_rutas"
CIUDADES_FILE = DATA_DIR / "ciudades.csv"
RUTAS_COSTES_FILE = DATA_DIR / "rutas_costes.csv"

# Valor de la hora del vendedor (EUR/h) usado como referencia por defecto.
# Es un parametro a estudiar: ver sensibilidad_valor_hora.py.
VALOR_HORA_POR_DEFECTO = 20.0


def cargar_ciudades(ruta: Path = CIUDADES_FILE):
    """Devuelve (lista_ciudades, DataFrame de coordenadas) en orden canonico."""
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    ciudades = df["ciudad"].astype(str).tolist()
    if len(ciudades) != 47 or len(set(ciudades)) != 47:
        raise ValueError(
            f"Se esperaban 47 ciudades unicas y hay {len(ciudades)} "
            f"({len(set(ciudades))} unicas)."
        )
    coords = df.set_index("ciudad")[["lat", "lon"]].astype(float)
    return ciudades, coords


class Costes:
    """Matrices dirigidas 47x47 y utilidades de coste para el TSP."""

    def __init__(
        self,
        ruta_costes: Path = RUTAS_COSTES_FILE,
        ruta_ciudades: Path = CIUDADES_FILE,
    ):
        self.ciudades, self.coords = cargar_ciudades(ruta_ciudades)
        self.n = len(self.ciudades)
        self.indice = {ciudad: i for i, ciudad in enumerate(self.ciudades)}

        df = pd.read_csv(ruta_costes, encoding="utf-8-sig")
        df = df[df["origin"].astype(str) != df["destination"].astype(str)]

        self.dist_km = self._rejilla(df, "distance_km")
        self.time_h = self._rejilla(df, "time_h")
        self.tax_eur = self._rejilla(df, "tax_eur")
        self.gas_eur = self._rejilla(df, "gas_eur")
        self.base_eur = self.tax_eur + self.gas_eur

    def _rejilla(self, df: pd.DataFrame, columna: str) -> np.ndarray:
        matriz = df.pivot(
            index="origin", columns="destination", values=columna
        )
        matriz = matriz.reindex(index=self.ciudades, columns=self.ciudades)
        valores = np.array(matriz.to_numpy(dtype=float), copy=True)
        fuera_diagonal = ~np.eye(self.n, dtype=bool)
        if np.isnan(valores[fuera_diagonal]).any():
            faltan = int(np.isnan(valores[fuera_diagonal]).sum())
            raise ValueError(
                f"Faltan {faltan} valores en la columna '{columna}': "
                "revisa rutas_costes.csv."
            )
        np.fill_diagonal(valores, 0.0)
        return valores

    def matriz_coste(self, valor_hora: float = VALOR_HORA_POR_DEFECTO) -> np.ndarray:
        """Matriz dirigida C = valor_hora*tiempo + tax + gas."""
        return valor_hora * self.time_h + self.base_eur

    def coste_tour(self, tour, valor_hora: float = VALOR_HORA_POR_DEFECTO) -> float:
        """Coste de un recorrido cerrado (incluye la arista ultimo->primero)."""
        idx = self._validar_tour(tour)
        return float(self.matriz_coste(valor_hora)[idx, np.roll(idx, -1)].sum())

    def desglose_tour(
        self, tour, valor_hora: float = VALOR_HORA_POR_DEFECTO
    ) -> dict:
        """Coste total y sus componentes para un recorrido cerrado."""
        idx = self._validar_tour(tour)
        nxt = np.roll(idx, -1)
        time_h = float(self.time_h[idx, nxt].sum())
        tax_eur = float(self.tax_eur[idx, nxt].sum())
        gas_eur = float(self.gas_eur[idx, nxt].sum())
        return {
            "coste_eur": valor_hora * time_h + tax_eur + gas_eur,
            "distancia_km": float(self.dist_km[idx, nxt].sum()),
            "tiempo_h": time_h,
            "tax_eur": tax_eur,
            "gas_eur": gas_eur,
            "valor_hora_eur": float(valor_hora),
        }

    def componentes_tour(self, tour):
        """(suma_tiempo, suma_base) del ciclo; util para la sensibilidad."""
        idx = self._validar_tour(tour)
        nxt = np.roll(idx, -1)
        return (float(self.time_h[idx, nxt].sum()),
                float(self.base_eur[idx, nxt].sum()))

    def tour_a_nombres(self, tour) -> list[str]:
        return [self.ciudades[i] for i in tour]

    def _validar_tour(self, tour) -> np.ndarray:
        idx = np.asarray(tour, dtype=int)
        if idx.ndim != 1 or idx.size != self.n:
            raise ValueError(
                f"El tour debe contener {self.n} ciudades; hay {idx.size}."
            )
        if sorted(idx.tolist()) != list(range(self.n)):
            raise ValueError(
                "El tour debe ser una permutacion exacta de 0..n-1."
            )
        return idx

    @staticmethod
    def firma(tour) -> tuple:
        """Firma invariante a rotacion para identificar un ciclo dirigido.

        No se normaliza la reflexion: en un TSP asimetrico A->B y B->A
        pueden tener coste distinto, por lo que recorrer el ciclo en sentido
        contrario es otra solucion.
        """
        idx = np.asarray(tour, dtype=int).tolist()
        n = len(idx)
        rotaciones = [
            tuple(idx[inicio:] + idx[:inicio]) for inicio in range(n)
        ]
        return min(rotaciones)

    @staticmethod
    def firma_a_tour(firma) -> np.ndarray:
        """Recupera un tour valido a partir de su firma (misma rotacion)."""
        return np.asarray(firma, dtype=int)


def cargar_costes() -> Costes:
    return Costes()


def _main():
    datos = cargar_costes()
    M = datos.matriz_coste(VALOR_HORA_POR_DEFECTO)
    print(f"Ciudades: {datos.n}")
    print(f"Matriz de coste: {M.shape} (dirigida)")
    print(f"Valor hora por defecto: {VALOR_HORA_POR_DEFECTO:.2f} EUR/h")
    print(f"Coste medio por arco: {M.mean():.2f} EUR "
          f"(min {M.min():.2f}, max {M.max():.2f})")
    print(f"Tiempo total de la matriz (h): {datos.time_h.sum():.1f}")
    print(f"Peajes valorados: {(datos.tax_eur > 0).sum()} arcos")
    tour = list(range(datos.n))
    desglose = datos.desglose_tour(tour, VALOR_HORA_POR_DEFECTO)
    print(f"Tour trivial (orden canonico) a 20 EUR/h: "
          f"{desglose['coste_eur']:.2f} EUR")


if __name__ == "__main__":
    _main()
