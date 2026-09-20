import { useEffect, useId, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import type { SearchHit } from "../lib/api";
import { useApi, useAsOf } from "../lib/hooks";

const KIND_LABEL: Record<SearchHit["kind"], string> = { ticker: "Ticker", person: "Person", fund: "Fund" };

export function Search() {
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const { asOf, search } = useAsOf();
  const navigate = useNavigate();
  const listId = useId();
  const root = useRef<HTMLDivElement>(null);

  // Wait for a pause in typing before asking, so a name is one request, not nine.
  useEffect(() => {
    const timer = window.setTimeout(() => setQuery(text.trim()), 160);
    return () => window.clearTimeout(timer);
  }, [text]);

  const { data } = useApi<SearchHit[]>(query.length >= 2 ? "/api/search" : null, { q: query, as_of: asOf });
  const hits = query.length >= 2 ? (data ?? []) : [];

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const go = (hit: SearchHit) => {
    setOpen(false);
    setText("");
    if (hit.kind === "ticker") navigate(`/t/${encodeURIComponent(hit.key)}${search}`);
    else {
      const params = new URLSearchParams(search);
      params.set("actor", hit.key);
      navigate(`/?${params}`);
    }
  };

  return (
    <div className="search" ref={root}>
      <label className="visually-hidden" htmlFor={`${listId}-input`}>
        Search a ticker, a person or a fund
      </label>
      <input
        id={`${listId}-input`}
        className="search__input"
        type="search"
        role="combobox"
        aria-expanded={open && hits.length > 0}
        aria-controls={listId}
        aria-autocomplete="list"
        autoComplete="off"
        placeholder="Search a ticker, a person or a fund"
        value={text}
        onChange={(event) => {
          setText(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((index) => Math.min(index + 1, hits.length - 1));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter") {
            const hit = hits[active];
            if (hit) go(hit);
            else if (/^[A-Za-z.\-]{1,6}$/.test(text.trim())) {
              navigate(`/t/${encodeURIComponent(text.trim().toUpperCase())}${search}`);
              setText("");
              setOpen(false);
            }
          } else if (event.key === "Escape") setOpen(false);
        }}
      />
      {open && query.length >= 2 && (
        <ul className="search__list" id={listId} role="listbox">
          {hits.length === 0 && <li className="search__empty">Nothing in the lake matches. Press Enter to open it as a ticker.</li>}
          {hits.map((hit, index) => (
            <li key={`${hit.kind}:${hit.key}`} role="option" aria-selected={index === active}>
              <button
                type="button"
                className={`search__hit${index === active ? " search__hit--active" : ""}`}
                onMouseEnter={() => setActive(index)}
                onClick={() => go(hit)}
              >
                <span className="search__hit-label">{hit.label}</span>
                <span className="search__hit-note">{hit.note ?? KIND_LABEL[hit.kind]}</span>
                <span className="search__hit-kind">{KIND_LABEL[hit.kind]}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
