# Description (Mar 13)

Old script that searches through the rules learned by `AnyBURL`, looking
for good rules with high support, high confidence and rule heads of the form
`(generic head, relation, constant tail)` whose relation-tail tuple make
up useful classes for the classifier.

Currently, not used as the classifier's classes are not derived from `AnyBURL`
but from the triples set directly.

Example output:

```
Rule(fires=1010, holds=992, confidence=0.9821782178217822, head=Fact(head='X', rel='/film/film/release_date_s./film/film_regional_release_date/film_release_distribution_medium', tail='/m/029j_'), body=[Fact(head='X', rel='/film/film/release_date_s./film/film_regional_release_date/film_release_distribution_medium', tail='A')])
Rule(fires=134, holds=130, confidence=0.9701492537313433, head=Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language', tail='/m/02h40lc'), body=[Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language', tail='A')])
Rule(fires=1393, holds=1315, confidence=0.9440057430007178, head=Fact(head='X', rel='/people/person/spouse_s./people/marriage/type_of_union', tail='/m/04ztj'), body=[Fact(head='X', rel='/people/person/spouse_s./people/marriage/type_of_union', tail='A')])
Rule(fires=235, holds=218, confidence=0.9276595744680851, head=Fact(head='X', rel='/tv/tv_program/languages', tail='/m/02h40lc'), body=[Fact(head='X', rel='/tv/tv_program/languages', tail='A')])
Rule(fires=1069, holds=970, confidence=0.9073900841908326, head=Fact(head='X', rel='/film/film/language', tail='/m/02h40lc'), body=[Fact(head='X', rel='/film/film/language', tail='A')])
Rule(fires=117, holds=103, confidence=0.8803418803418803, head=Fact(head='X', rel='/location/statistical_region/gdp_nominal./measurement_unit/dated_money_value/currency', tail='/m/09nqf'), body=[Fact(head='/m/085h1', rel='/user/ktrueman/default_domain/international_organization/member_states', tail='X')])
Rule(fires=130, holds=114, confidence=0.8769230769230769, head=Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language', tail='/m/02h40lc'), body=[Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location', tail='A')])
Rule(fires=111, holds=97, confidence=0.8738738738738738, head=Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language', tail='/m/02h40lc'), body=[Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/contact_category', tail='A')])
Rule(fires=117, holds=102, confidence=0.8717948717948718, head=Fact(head='X', rel='/film/film/genre', tail='/m/05p553'), body=[Fact(head='X', rel='/film/film/genre', tail='/m/06cvj')])
Rule(fires=101, holds=88, confidence=0.8712871287128713, head=Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language', tail='/m/02h40lc'), body=[Fact(head='X', rel='/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location', tail='/m/09c7w0')])
287419 rules
3381 good rules
```

# Data directory

```
data/
    anyburl/
        rules/
            FB15-237/                       # Input AnyBURL Rules Directory
                alpha.stdout
                alpha-10
                alpha-50
                alpha-100
                alpha_log
                config-learn.properties
```
