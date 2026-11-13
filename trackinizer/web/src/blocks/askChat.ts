// THE ANSWERING SIDE OF THE ASK BAR.
//
// `ui/ask.tsx` has rendered an answer under the question since it was written — a `q`
// line, an `a` line, a `thinking…` while it waits. Nothing ever filled it in: the bar
// fired the question at the chat column and kept a handle to a reply no one wrote, so
// the box cleared on Enter and the reader was left looking at the page they asked about.
//
// This is the writer it was waiting for. The question goes to the station's own chat —
// the same conversation the column shows, not a second channel — and the reply is read
// back off the partial stream and handed to the bar. One question, one answer, in place.
import { getChatHead, readChatPartial, sendChatLine } from "../api/chats";

/** How long to wait for the assistant to open the conversation before giving up. */
const OPEN_TIMEOUT_MS = 30_000;
/** The whole exchange's ceiling: a follow-up that runs longer belongs in the column. */
const REPLY_TIMEOUT_MS = 180_000;

/** A conversation to ask in: the station, and the thread it already has (null starts one). */
export type AskTarget = {
  readonly workspaceId: string;
  readonly conversationId: string | null;
  /** The route the asker is on, which the server records with the line. */
  readonly page: string | null;
};

/**
 * Ask `question` in the station's chat and return the assistant's reply.
 *
 * Resolves with the text; throws with a sentence the bar can print. The caller decides
 * what to do with a new conversation id — a station that had none now has one, and the
 * chat column shows the same thread, because this deliberately does not open its own.
 */
export async function askTheChat(
  target: AskTarget,
  question: string,
  { signal }: { readonly signal?: AbortSignal } = {},
): Promise<{ readonly text: string; readonly conversationId: string }> {
  const sent = await sendChatLine(
    {
      workspaceId: target.workspaceId,
      text: question,
      conversationId: target.conversationId,
      page: target.page,
      trail: [],
    },
    crypto.randomUUID(),
  );

  // The session is there at once when the thread is already open; a new conversation has
  // none until the assistant picks it up, and the server answers 404 the whole time.
  let sessionId = sent.session_id ?? null;
  const openBy = Date.now() + OPEN_TIMEOUT_MS;
  while (sessionId === null) {
    if (signal?.aborted) throw new Error("the question was withdrawn");
    if (Date.now() > openBy) throw new Error("the assistant did not pick the question up");
    await sleep(500, signal);
    const head = await getChatHead(sent.conversation_id, { signal });
    sessionId = head?.session_id ?? null;
  }

  // `readChatPartial` holds for up to 25 s when there is nothing new, so this is a
  // long-poll and not a spin: each pass either returns a longer answer or costs one wait.
  let after = 0;
  let text = "";
  const replyBy = Date.now() + REPLY_TIMEOUT_MS;
  for (;;) {
    if (signal?.aborted) throw new Error("the question was withdrawn");
    if (Date.now() > replyBy) {
      // Partial text is still an answer: show what arrived rather than throwing it away.
      if (text.trim() !== "") break;
      throw new Error("the assistant did not answer in time");
    }
    const state = await readChatPartial(sessionId, after, { signal });
    after = state.seq;
    if (state.partial) {
      text = state.partial.text;
      if (state.partial.done) break;
    } else if (text.trim() !== "") {
      // The partial is cleared once the finished text is stored, so a reply that was
      // streaming and is now gone is a reply that landed.
      break;
    }
  }

  if (text.trim() === "") throw new Error("the assistant answered with nothing");
  return { text: text.trim(), conversationId: sent.conversation_id };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new Error("aborted")); }, { once: true });
  });
}
