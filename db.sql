create table chat_history (
  id bigserial primary key,
  user_id text not null,
  role text not null, -- 'user' or 'assistant'
  content text not null,
  timestamp timestamptz not null default now()
);

-- Optional: Index for faster queries by user
create index idx_chat_history_user_id on chat_history(user_id);