-- Migration 010: learning over time. ADDITIVE ONLY. Safe to run on an existing database.
-- Extra face templates per student, taken from attendance sessions the teacher confirmed.
-- Matching uses each student's closest template, so new hairstyles, glasses or lighting are
-- recognised over time. The app works without this table (learning is simply switched off).

create table if not exists face_templates (
  template_id  bigint generated always as identity primary key,
  student_id   bigint not null references students(student_id) on delete cascade,  -- profile deletion removes them
  embedding    jsonb not null,                                                       -- 128 floats (dlib)
  source       text not null default 'confirmed_attendance' check (source in ('confirmed_attendance')),
  subject_id   bigint references subjects(subject_id) on delete set null,
  added_by     bigint references teachers(teacher_id) on delete set null,
  created_at   timestamptz not null default now()
);
create index if not exists face_templates_student_idx on face_templates(student_id, template_id);

alter table face_templates enable row level security;          -- no policies: server key only
revoke all on table face_templates from anon, authenticated;
