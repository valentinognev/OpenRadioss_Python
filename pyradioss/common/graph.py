"""
Graph data structures, connected components, and cycle detection.

Fortran / C++ source citations:
- Graph.hpp:      $OR_SRC/common_source/tools/graphs/Graph.hpp
- Graph.cpp:      $OR_SRC/common_source/tools/graphs/Graph.cpp
- Graph_api.cpp:  $OR_SRC/common_source/tools/graphs/Graph_api.cpp
- dsgraph_mod.F:  $OR_SRC/engine/share/modules/dsgraph_mod.F

Theory notes:
- Used in OpenRadioss for surface decomposition, contact search partitioning,
  mesh connectivity analysis, and topological cycle tracing.
- Represents an undirected graph with N vertices and M edge connections.
- Identifies connected components using Depth-First Search (DFS).
- Detects whether connected components form closed 2-regular simple cycles,
  and orders the vertices along the cycle boundary.
"""

from __future__ import annotations

import numpy as np


class Graph:
    """Undirected graph for connectivity and cycle decomposition.

    Ported from $OR_SRC/common_source/tools/graphs/Graph.cpp

    Parameters
    ----------
    npt : int
        Number of points / vertices (0-indexed: 0, 1, ..., npt - 1).
    nconnect : int
        Number of edge connections.
    connect_list : list of int or sequence of pairs
        Flattened or 2D array of connections [p1_0, p2_0, p1_1, p2_1, ...].
    """

    def __init__(
        self,
        npt: int,
        nconnect: int | None = None,
        connect_list: list[int] | np.ndarray | None = None,
    ):
        self.m_npt: int = int(npt)
        self.m_adj_list: list[list[int]] = [[] for _ in range(self.m_npt)]
        self.m_degree: list[int] = [0] * self.m_npt
        self.m_nb_connected_components: int = 0
        self.m_total_size: int = 0
        self.m_path: list[list[int]] = []
        self.m_path_diag: list[list[int]] = []
        self.m_color: list[int] = [0] * self.m_npt

        if connect_list is not None:
            arr = np.asarray(connect_list, dtype=np.int64).reshape(-1)
            num_edges = len(arr) // 2 if nconnect is None else nconnect
            self.m_nconnect: int = num_edges

            for i in range(num_edges):
                p1 = int(arr[2 * i])
                p2 = int(arr[2 * i + 1])
                if 0 <= p1 < self.m_npt and 0 <= p2 < self.m_npt:
                    self.m_adj_list[p1].append(p2)
                    self.m_adj_list[p2].append(p1)

            for i in range(self.m_npt):
                # Sort and remove duplicates matching Graph.cpp lines 36-39
                self.m_adj_list[i] = sorted(list(set(self.m_adj_list[i])))
                self.m_degree[i] = len(self.m_adj_list[i])
        else:
            self.m_nconnect = 0

    def dfs(self, p0: int, path: list[int]) -> list[int]:
        """Depth-First Search traversal starting from vertex p0.

        Ported from Graph::dfs in Graph.cpp (lines 45-72).
        Color coding: 0 = white (unvisited), 1 = gray (visiting), 2 = black (visited).
        """
        res = [-2] * self.m_npt
        p: list[int] = []

        p.append(p0)
        path.append(p0)
        self.m_color[p0] = 1
        res[p0] = -1

        ok = True
        while ok:
            si = p[-1]
            # Find first neighbor with color == 0
            found_neighbor = None
            for neighbor in self.m_adj_list[si]:
                if self.m_color[neighbor] == 0:
                    found_neighbor = neighbor
                    break

            if found_neighbor is not None:
                sj = found_neighbor
                p.append(sj)
                self.m_color[sj] = 1
                path.append(sj)
                res[sj] = si
            else:
                self.m_color[si] = 2
                p.pop()

            ok = len(p) > 0

        return res

    def build_path(self) -> None:
        """Find connected components and build paths through each component.

        Ported from Graph::build_path in Graph.cpp (lines 74-106).
        """
        self.m_nb_connected_components = 0
        self.m_path = []
        self.m_path_diag = []
        self.m_color = [0] * self.m_npt

        ok = self.m_npt > 0
        s0 = 0
        while ok:
            self.m_nb_connected_components += 1
            path: list[int] = []
            diag = self.dfs(s0, path)
            self.m_path_diag.append(diag)
            self.m_path.append(path)

            # Find next unvisited vertex (color == 0)
            try:
                s0 = self.m_color.index(0)
                ok = True
            except ValueError:
                ok = False

        self.m_total_size = sum(len(comp) for comp in self.m_path)

    def build_cycle(self) -> list[bool]:
        """Check if components are simple closed 2-regular cycles and order them.

        Ported from Graph::build_cycle in Graph.cpp (lines 108-143).

        Returns
        -------
        is_cycle : list of bool
            Boolean flag per connected component indicating whether it forms a simple cycle.
        """
        if self.m_nb_connected_components == 0:
            self.build_path()

        res = [False] * self.m_nb_connected_components
        for iconnect in range(self.m_nb_connected_components):
            comp_path = self.m_path[iconnect]
            if len(comp_path) < 3:
                continue

            degrees = [self.m_degree[node] for node in comp_path]
            max_deg = max(degrees)
            min_deg = min(degrees)

            # 2-regular graph (every vertex has degree 2)
            if max_deg == 2 and min_deg == 2:
                path_new: list[int] = []
                s0 = comp_path[0]
                sinit = s0
                s1 = self.m_adj_list[s0][0]

                ok = True
                while ok:
                    path_new.append(s0)
                    adj1 = self.m_adj_list[s1]
                    s2 = adj1[1] if adj1[0] == s0 else adj1[0]

                    ok = len(path_new) != len(comp_path)
                    if not ok and s1 == sinit:
                        res[iconnect] = True

                    s0 = s1
                    s1 = s2

                if res[iconnect]:
                    self.m_path[iconnect] = path_new

        return res

    def get_nb_connected_components(self) -> int:
        return self.m_nb_connected_components

    def get_path(self) -> list[list[int]]:
        return self.m_path

    def get_sizes(self) -> list[int]:
        return [len(comp) for comp in self.m_path]

    def get_total_size(self) -> int:
        return self.m_total_size

    def get_adj_list(self) -> list[list[int]]:
        return self.m_adj_list


def connected_components(
    npt: int, connections: list[tuple[int, int]] | np.ndarray
) -> list[list[int]]:
    """Compute connected components of an undirected graph.

    Convenience wrapper around Graph.build_path.

    Parameters
    ----------
    npt : int
        Number of vertices (0 to npt-1).
    connections : list of tuple(int, int) or array of shape (M, 2)
        Edge connections.

    Returns
    -------
    components : list of list of int
        List of vertex indices in each connected component.
    """
    g = Graph(npt, connect_list=connections)
    g.build_path()
    return g.get_path()


def find_cycles(
    npt: int, connections: list[tuple[int, int]] | np.ndarray
) -> tuple[list[bool], list[list[int]]]:
    """Identify closed simple cycles in the graph.

    Convenience wrapper around Graph.build_cycle.

    Parameters
    ----------
    npt : int
        Number of vertices (0 to npt-1).
    connections : list of tuple(int, int) or array of shape (M, 2)
        Edge connections.

    Returns
    -------
    is_cycle : list of bool
        Flags indicating which components form simple cycles.
    cycle_paths : list of list of int
        Component paths (ordered along the cycle perimeter for cycle components).
    """
    g = Graph(npt, connect_list=connections)
    is_cycle = g.build_cycle()
    return is_cycle, g.get_path()
