# Platform Team Meeting — 9 October 2026

**Facilitator:** Priya N. · **Note taker:** Tomás R. · **Duration:** 60 minutes

## Attendees

- Priya N. (engineering manager)
- Tomás R. (backend)
- Wen L. (infrastructure)
- Amara O. (product)
- Jonas K. (security)

## Agenda

1. Quarterly budget review
2. Hiring update
3. Database migration plan
4. Any other business

## Discussion

### Budget review

Amara presented the quarterly numbers. Cloud spending was **$48,200**, which is 12% below
the forecast because the staging cluster was shut down at night. The team agreed to keep
the nightly shutdown and to move the savings into the load-testing budget.

### Hiring update

Two backend positions are open. Seven candidates reached the final interview round, and
one offer was accepted: the new backend engineer starts on 2 November. The second
position stays open until the end of the year.

### Database migration

Wen proposed moving the orders database from PostgreSQL 13 to PostgreSQL 16. The
migration will use logical replication, so the old and new clusters run side by side for
two weeks. The cut-over window is planned for **Saturday 14 November, 02:00 to 04:00
UTC**, when traffic is lowest.

Risks discussed:

- Extensions that are not available on version 16
- Replication lag during the nightly batch jobs
- Connection strings hard-coded in older services

### Any other business

Jonas reminded everyone that the annual security training must be finished by the end of
the month. Laptops without disk encryption will lose network access on 1 December.

## Decisions

- Keep the nightly shutdown of the staging cluster.
- Approve the PostgreSQL 16 migration with the November cut-over window.
- Freeze feature releases during the week of the migration.

## Action items

- [ ] Wen: write the rollback runbook for the migration by 30 October
- [ ] Tomás: list all services with hard-coded connection strings by 23 October
- [ ] Amara: update the quarterly budget sheet by 16 October
- [x] Priya: send the offer letter to the backend candidate

## Next meeting

The next meeting is on 23 October 2026 at 10:00 in room B-214.
