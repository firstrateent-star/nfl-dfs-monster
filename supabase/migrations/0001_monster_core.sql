-- Monster core schema. Private by default; market tables are downstream only.
create schema if not exists monster;
revoke all on schema monster from anon, authenticated;

create table if not exists monster.sources (
  source_id bigint generated always as identity primary key,
  source_name text not null,
  dataset_type text not null,
  uri text,
  sha256 text not null unique,
  effective_at timestamptz,
  ingested_at timestamptz not null default now(),
  row_count bigint,
  license_note text,
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists monster.players (
  player_id text primary key,
  display_name text not null,
  position text,
  current_team text,
  birth_date date,
  birth_time time,
  birth_time_confidence numeric(5,4),
  birth_place text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists monster.teams (
  team_id text primary key,
  team_name text,
  conference text,
  division text,
  updated_at timestamptz not null default now()
);

create table if not exists monster.slates (
  slate_id text primary key,
  season integer not null,
  week integer not null,
  slate_date date not null,
  site text,
  contest_id text,
  created_at timestamptz not null default now()
);

create table if not exists monster.games (
  game_id text primary key,
  season integer not null,
  week integer not null,
  away_team text not null references monster.teams(team_id),
  home_team text not null references monster.teams(team_id),
  kickoff_at timestamptz,
  stadium_name text,
  dome_flag boolean,
  latitude double precision,
  longitude double precision,
  updated_at timestamptz not null default now()
);

create table if not exists monster.slate_games (
  slate_id text not null references monster.slates(slate_id) on delete cascade,
  game_id text not null references monster.games(game_id) on delete cascade,
  primary key (slate_id, game_id)
);

create table if not exists monster.slate_players (
  slate_id text not null references monster.slates(slate_id) on delete cascade,
  player_id text not null references monster.players(player_id),
  game_id text references monster.games(game_id),
  team_id text references monster.teams(team_id),
  opponent_id text references monster.teams(team_id),
  salary integer,
  roster_position text,
  injury_status text,
  source_id bigint references monster.sources(source_id),
  primary key (slate_id, player_id)
);

create table if not exists monster.feature_definitions (
  feature_name text primary key,
  entity_type text not null check (entity_type in ('player','team','game','unit','slate')),
  family text not null,
  cadence text,
  jurisdiction text[] not null default '{}',
  authority numeric(6,5) not null default 0,
  production_weight numeric(6,5),
  market_derived boolean not null default false,
  promotion_gate text,
  notes text
);

create table if not exists monster.feature_values (
  feature_value_id bigint generated always as identity primary key,
  entity_type text not null,
  entity_id text not null,
  feature_name text not null references monster.feature_definitions(feature_name),
  value_num double precision,
  value_text text,
  unit text,
  effective_at timestamptz not null,
  source_id bigint references monster.sources(source_id),
  run_id text,
  confidence numeric(6,5),
  metadata jsonb not null default '{}'::jsonb,
  unique(entity_type, entity_id, feature_name, effective_at, source_id)
);

create table if not exists monster.model_runs (
  run_id text primary key,
  model_version text not null,
  git_commit text,
  parent_run_id text,
  slate_id text references monster.slates(slate_id),
  market_blind boolean not null default true,
  seed bigint,
  worlds integer,
  status text not null,
  created_at timestamptz not null default now(),
  input_hash text,
  config_hash text,
  notes text
);

create table if not exists monster.game_summaries (
  run_id text not null references monster.model_runs(run_id) on delete cascade,
  game_id text not null references monster.games(game_id),
  away_mean double precision,
  home_mean double precision,
  total_mean double precision,
  p10 double precision,
  p50 double precision,
  p90 double precision,
  p60_plus double precision,
  primary key (run_id, game_id)
);

create table if not exists monster.team_summaries (
  run_id text not null references monster.model_runs(run_id) on delete cascade,
  team_id text not null references monster.teams(team_id),
  game_id text not null references monster.games(game_id),
  mean double precision,
  p20 double precision,
  p50 double precision,
  p90 double precision,
  p95 double precision,
  p30_plus double precision,
  p35_plus double precision,
  p40_plus double precision,
  primary key (run_id, team_id)
);

create table if not exists monster.player_summaries (
  run_id text not null references monster.model_runs(run_id) on delete cascade,
  player_id text not null references monster.players(player_id),
  game_id text references monster.games(game_id),
  team_id text references monster.teams(team_id),
  mean double precision,
  p20 double precision,
  p50 double precision,
  p75 double precision,
  p90 double precision,
  p95 double precision,
  p99 double precision,
  active_probability double precision,
  role_entropy double precision,
  allocation_share_mean double precision,
  primary key (run_id, player_id)
);

create table if not exists monster.artifacts (
  artifact_id bigint generated always as identity primary key,
  run_id text not null references monster.model_runs(run_id) on delete cascade,
  artifact_type text not null,
  storage_path text not null,
  sha256 text not null,
  bytes bigint,
  row_count bigint,
  column_count integer,
  retention_class text not null check(retention_class in ('canonical','rolling','ephemeral')),
  created_at timestamptz not null default now(),
  unique(run_id, artifact_type, storage_path)
);

-- Downstream only. The blind simulator must not query these tables.
create table if not exists monster.market_lines (
  slate_id text not null references monster.slates(slate_id),
  game_id text not null references monster.games(game_id),
  observed_at timestamptz not null,
  total double precision,
  away_spread double precision,
  away_implied double precision,
  home_implied double precision,
  source_id bigint references monster.sources(source_id),
  primary key (slate_id, game_id, observed_at)
);

create table if not exists monster.dfs_prices (
  slate_id text not null references monster.slates(slate_id),
  site text not null,
  player_id text not null references monster.players(player_id),
  salary integer not null,
  roster_position text,
  external_player_id text,
  source_id bigint references monster.sources(source_id),
  primary key (slate_id, site, player_id)
);

create table if not exists monster.audit_events (
  audit_id bigint generated always as identity primary key,
  run_id text references monster.model_runs(run_id),
  audit_type text not null,
  result text not null,
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists monster.dependency_rules (
  trigger_type text not null,
  trigger_scope text not null,
  downstream_stage text not null,
  propagation_order integer not null,
  rerun_scope text not null,
  notes text,
  primary key(trigger_type, trigger_scope, downstream_stage)
);

create table if not exists monster.dirty_queue (
  dirty_id bigint generated always as identity primary key,
  trigger_type text not null,
  entity_type text,
  entity_key text,
  source_id bigint references monster.sources(source_id),
  detected_at timestamptz not null default now(),
  downstream_stage text not null,
  status text not null default 'pending',
  notes text
);

revoke all on all tables in schema monster from anon, authenticated;
alter default privileges in schema monster revoke all on tables from anon, authenticated;
