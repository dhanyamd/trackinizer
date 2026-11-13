import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { Askable, type Asked } from "./ask";

afterEach(cleanup);

/** A composition's item, as `compose.tsx` marks one up for the bar to point at. */
function Surface() {
  return <div data-node="figure" data-item='{"kind":"chart"}'>a figure</div>;
}

/** Point at the item, type `question`, and submit — the three steps a reader takes. */
function ask(question: string) {
  fireEvent.click(screen.getByText("a figure"));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: question } });
  fireEvent.submit(screen.getByRole("textbox"));
}

test("an answer returned by the asker is shown under the question, where it was asked", async () => {
  render(<Askable onAsk={() => Promise.resolve("2.59% of resident bytes, at batch 1.")}><Surface /></Askable>);
  ask("what share is the cache?");

  // The question stands while the answer is in flight: a bar that cleared itself would
  // leave the reader looking at the page they asked about with nothing to show for it.
  expect(screen.getByText("thinking…")).toBeTruthy();
  expect(await screen.findByText("2.59% of resident bytes, at batch 1.")).toBeTruthy();
  expect(screen.getByText("what share is the cache?")).toBeTruthy();
  expect(screen.queryByText("thinking…")).toBeNull();
});

test("a question that cannot be answered says so in the bar rather than hanging on 'thinking'", async () => {
  render(<Askable onAsk={() => Promise.reject(new Error("this tab is not on a station yet"))}><Surface /></Askable>);
  ask("why?");
  expect(await screen.findByText("Not answered: this tab is not on a station yet.")).toBeTruthy();
});

test("the answer to a question the reader has moved on from does not overwrite the one on screen", async () => {
  const replies: ((text: string) => void)[] = [];
  render(
    <Askable onAsk={() => new Promise<string>((resolve) => { replies.push(resolve); })}><Surface /></Askable>,
  );
  ask("first");
  ask("second");
  // The first question's answer lands late. The bar shows the second, so the first is
  // dropped: an answer under the wrong question is worse than no answer.
  await act(async () => { replies[0]!("late answer to the first"); });
  expect(screen.queryByText("late answer to the first")).toBeNull();
  await act(async () => { replies[1]!("the answer to the second"); });
  expect(await screen.findByText("the answer to the second")).toBeTruthy();
});

test("an asker that answers nothing leaves the store path intact, and a reply written there shows", async () => {
  localStorage.clear();
  const onAsk = vi.fn<(asked: Asked) => void>();
  render(<Askable onAsk={onAsk}><Surface /></Askable>);
  ask("answered by something else");

  // No promise came back, so the question is filed for whatever writes replies beside it.
  const [[asked]] = onAsk.mock.calls;
  expect(asked.question).toBe("answered by something else");
  const key = Object.keys(localStorage).find((k) => k.endsWith(":q"));
  expect(key).toBeTruthy();
  expect(JSON.parse(localStorage.getItem(key!)!).node).toBe("figure");

  localStorage.setItem(key!.replace(/:q$/, ":a"), "written by the other side");
  expect(await screen.findByText("written by the other side", undefined, { timeout: 3000 })).toBeTruthy();
});
