"""Generación de secuencias (backtracking) con poda por invariantes.

Uso previsto: explorar secuencias de señales/órdenes para comprobar
que una FSM (o lógica de enclavamiento) respeta invariantes de seguridad.
La poda corta ramas que violan el invariante, evitando explosión combinatoria.
"""
from typing import Callable, List, Dict, Tuple, Iterable

State = str
Event = str

class FSM:
    def __init__(self, transitions: Dict[Tuple[State, Event], State], start: State):
        self.T = transitions
        self.start = start

    def step(self, state: State, event: Event) -> State | None:
        return self.T.get((state, event))

def generate_sequences(
    fsm: FSM,
    alphabet: Iterable[Event],
    length: int,
    invariant_ok: Callable[[List[State], List[Event]], bool],
    limit: int | None = None,
) -> List[List[Event]]:
    """Devuelve secuencias de eventos que respetan el invariante.

    - fsm: autómata determinista (parcial) con step(s, e)->s'|None
    - alphabet: eventos disponibles
    - length: longitud objetivo
    - invariant_ok(path_states, path_events): predicado de poda
    - limit: máximo número de secuencias devueltas (opcional)
    """
    solutions: List[List[Event]] = []
    path_states: List[State] = [fsm.start]
    path_events:  List[Event] = []

    def dfs(depth: int):
        nonlocal solutions
        if limit is not None and len(solutions) >= limit:
            return
        if depth == length:
            if invariant_ok(path_states, path_events):
                solutions.append(list(path_events))
            return
        s = path_states[-1]
        for e in alphabet:
            s2 = fsm.step(s, e)
            if s2 is None:
                continue
            path_events.append(e)
            path_states.append(s2)
            if invariant_ok(path_states, path_events):
                dfs(depth+1)
            path_states.pop()
            path_events.pop()
    dfs(0)
    return solutions

if __name__ == "__main__":
    # Ejemplo mínimo con invariantes
    T = {
        ("IDLE", "start"): "RUN",
        ("RUN", "stop"): "IDLE",
        ("RUN", "alarm"): "SAFE",
        ("SAFE", "reset"): "IDLE",
    }
    fsm = FSM(T, "IDLE")
    # invariante: no se permite RUN->reset directo (requiere stop o alarm)
    def invariant_ok(states: List[State], events: List[Event]) -> bool:
        if len(events) >= 1 and events[-1] == "reset":
            # debe venir de SAFE
            return states[-2] == "SAFE"
        return True
    seqs = generate_sequences(fsm, ["start","stop","alarm","reset"], length=3, invariant_ok=invariant_ok, limit=10)
    print("ejemplos_validos=", seqs)
