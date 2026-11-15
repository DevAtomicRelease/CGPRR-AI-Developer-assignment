# About this dataset (SYNTHETIC)

This dataset was generated for the NEET.info demo assistant by
`scripts/make_synthetic_data.py` with a fixed random seed. All college names,
ranks, seat counts and fees are fictional. It contains no real student,
candidate, doctor or college records.

## Coverage

- **Years:** 2023, 2024 and 2025 counselling. There is no data for 2026 or for
  any year before 2023.
- **Rounds:** Round 1 and Round 2 only. Mop-up and stray rounds are not covered.
- **Courses:** MBBS (all 22 colleges) and BDS (6 colleges). BAMS, BHMS, BSc
  Nursing and other courses are not covered.
- **Quotas:** All India Quota, State Quota and Deemed University. NRI,
  management and PwD quotas are not covered.
- **Categories:** GEN, EWS, OBC, SC and ST.
- **States:** West Bengal, Odisha, Bihar, Jharkhand, Assam, Delhi, Maharashtra,
  Karnataka, Tamil Nadu, Uttar Pradesh and Madhya Pradesh.
- **Fees:** annual tuition and hostel fees for 2025 only.

## Known gaps

Some ST rows are deliberately missing (no allotment recorded in that year,
round and quota). When a row is missing, the correct answer is that the figure
is not in the sources; the assistant must not estimate it.

## Files

- `tables/closing_ranks.csv`: opening and closing NEET All India Rank by year,
  round, college, course, quota and category.
- `tables/seat_matrix.csv`: number of seats by year, college, course, quota
  and category.
- `tables/fees_2025.csv`: annual tuition and hostel fee by college and course.
- `docs/counselling_handbook.md`: simplified synthetic counselling rules.
- `docs/college_profiles.md`: short synthetic profile of each college.
