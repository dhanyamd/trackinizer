import { useSyncExternalStore } from "react";
import type { FocusRef } from "../router/route";
import { namedRows } from "../visuals/chatRefs";

/**
 * What the canvas is about, so the rail's Graph can open on it.
 *
 * The graph is one view of 5,000-odd records. Opened bare it is a hairball, and a
 * reader who has just read an answer about one Belief has to find that Belief in it
 * by hand — the answer and the graph of the thing the answer is about were two
 * unconnected places. A detail page has had `Show in graph` for this all along
 * (`detail/index.tsx`); this is the same focus, read off the composition instead of
 * a row, so the rail's Graph lands lit on what was under discussion.
 *
 * A focus lights and does not filter: the whole graph still draws, with what lies
 * within the hops lit and the rest dimmed. So guessing the topic wrongly costs a
 * reader nothing they had — the worst case is the view they would have got anyway.
 */
let topic: FocusRef | null = null;
const watchers = new Set<() => void>();

/** Say what the canvas is about now, or `null` when it is about nothing nameable. */
export function setTopic(next: FocusRef | null): void {
  if (sameTopic(topic, next)) return;
  topic = next;
  for (const watcher of [...watchers]) watcher();
}

/** What the canvas is about, outside React. */
export function currentTopic(): FocusRef | null {
  return topic;
}

/** What the canvas is about, as a view reads it. */
export function useTopic(): FocusRef | null {
  return useSyncExternalStore(subscribe, currentTopic, nothing);
}

/**
 * The row `words` is about: the first one it names, by the rule the reader sees.
 *
 * `namedRows` reads what the page renders as a ref — a `Kind#seq` of a kind the app
 * knows, or a UUID — and skips the ones inside code, so a composition that quotes
 * `Issue#7` in a snippet is not about Issue#7. First rather than most-named: an
 * answer leads with its subject, and counting would let a long tail of citations
 * outvote the title.
 */
export function topicOf(words: string, kinds: readonly string[]): FocusRef | null {
  return namedRows(words, kinds)[0] ?? null;
}

/** How a focus on `ref` reads in a title. */
export function topicName(ref: FocusRef): string {
  return "id" in ref ? ref.id.slice(0, 8) : `${ref.kind}#${ref.seq}`;
}

/** Whether `a` and `b` name the same row. */
export function sameTopic(a: FocusRef | null, b: FocusRef | null): boolean {
  if (a === null || b === null) return a === b;
  if ("id" in a || "id" in b) return "id" in a && "id" in b && a.id === b.id;
  return a.kind === b.kind && a.seq === b.seq;
}

function subscribe(watcher: () => void): () => void {
  watchers.add(watcher);
  return () => {
    watchers.delete(watcher);
  };
}

/** The server renders no canvas, so it is about nothing. */
function nothing(): null {
  return null;
}
