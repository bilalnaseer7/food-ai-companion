# Personalization vs Filter Bubble Analysis

This analysis supports Bilal's Milestone 3 contribution. It uses the cleaned local restaurant metadata to compare a profile-personalized recommendation proxy against the same candidate pool with a diversity constraint.

Important limitation: this is a deterministic metadata analysis, not a human evaluation and not an LLM quality score. The accepted-history set is simulated from each profile's strongest local matches so the filter-bubble metric can be computed reproducibly without inventing user survey scores.

## Metric Definitions

- `name_overlap`: fraction of current recommendations that exactly repeat simulated accepted restaurants.
- `category_overlap`: fraction of current primary cuisine/category labels also present in accepted history.
- `novelty`: `1 - name_overlap`; higher means fewer repeated restaurant names.
- `category_diversity`: unique primary categories divided by recommendation count.
- `category_entropy`: normalized spread of primary categories.
- `filter_bubble_index`: weighted repetition-risk estimate using name overlap, category overlap, and category concentration.
- `profile_alignment`: fraction of recommendations matching preferred cuisines/liked foods while avoiding disliked foods.

## Average Metrics by Mode

| Mode | Filter bubble index | Novelty | Category diversity | Profile alignment |
|---|---:|---:|---:|---:|
| profile_proxy | 0.8311 | 0.1778 | 0.4000 | 1.0000 |
| profile_proxy_diversity_constraint | 0.2700 | 1.0000 | 0.6000 | 1.0000 |

## Recommendation Sets

## italian_pasta - I want something good for dinner in NYC.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Olio e Piu | Italian | pizza | 0.98 |
| 2 | Bleecker Street Pizza | Italian | pizza | 0.98 |
| 3 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.98 |
| 4 | The Marshal | Italian | pizza | 0.98 |
| 5 | Trattoria Trecolori | Italian | pizza | 0.98 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Olio e Piu | Italian | pizza | 1.03 |
| 2 | Bleecker Street Pizza | Italian | pizza | 1.03 |
| 3 | Tony's Di Napoli - Midtown | Italian | Pasta | 1.03 |
| 4 | The Marshal | Italian | pizza | 1.03 |
| 5 | Trattoria Trecolori | Italian | pizza | 1.03 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Stage Door Deli | Pizza | pizza | 0.9659 |
| 2 | Capizzi | Italian | cannoli | 1.03 |
| 3 | La Masseria | Italian | pizza | 1.03 |
| 4 | Don Antonio | Italian | pizza | 1.03 |
| 5 | Da Marino Restaurant | Italian | Pasta | 1.03 |

## italian_pasta - Find a casual but memorable place for tonight.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Olio e Piu | Italian | pizza | 0.98 |
| 2 | Bleecker Street Pizza | Italian | pizza | 0.98 |
| 3 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.98 |
| 4 | The Marshal | Italian | pizza | 0.98 |
| 5 | Trattoria Trecolori | Italian | pizza | 0.98 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Olio e Piu | Italian | pizza | 1.03 |
| 2 | Bleecker Street Pizza | Italian | pizza | 1.03 |
| 3 | Tony's Di Napoli - Midtown | Italian | Pasta | 1.03 |
| 4 | The Marshal | Italian | pizza | 1.03 |
| 5 | Trattoria Trecolori | Italian | pizza | 1.03 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Stage Door Deli | Pizza | pizza | 0.9659 |
| 2 | Capizzi | Italian | cannoli | 1.03 |
| 3 | La Masseria | Italian | pizza | 1.03 |
| 4 | Don Antonio | Italian | pizza | 1.03 |
| 5 | Da Marino Restaurant | Italian | Pasta | 1.03 |

## italian_pasta - Recommend something flavorful but not too expensive.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Olio e Piu | Italian | pizza | 0.98 |
| 2 | Bleecker Street Pizza | Italian | pizza | 0.98 |
| 3 | Tony's Di Napoli - Midtown | Italian | Pasta | 0.98 |
| 4 | The Marshal | Italian | pizza | 0.98 |
| 5 | Trattoria Trecolori | Italian | pizza | 0.98 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Babbo Ristorante e Enoteca | Italian | pizza | 1.03 |
| 2 | Da Gennaro Restaurant | Italian | pizza | 1.03 |
| 3 | Ribalta Pizza | Italian | pizza | 0.9903 |
| 4 | Lattanzi Ristorante | Italian | pizza | 0.9839 |
| 5 | Il Gattopardo | Italian | pasta | 0.9817 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Kiss My Slice | Pizza | pizza | 0.9381 |
| 2 | Babbo Ristorante e Enoteca | Italian | pizza | 1.03 |
| 3 | Da Gennaro Restaurant | Italian | pizza | 1.03 |
| 4 | Ribalta Pizza | Italian | pizza | 0.9903 |
| 5 | Lattanzi Ristorante | Italian | pizza | 0.9839 |

## japanese_date_night - I want something good for dinner in NYC.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 0.98 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 0.98 |
| 3 | Natsumi | Japanese | sushi | 0.9393 |
| 4 | Zuma | Japanese | sushi | 0.9345 |
| 5 | The Breslin | Japanese | sushi | 0.9304 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 1.03 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 1.03 |
| 3 | Natsumi | Japanese | sushi | 0.9893 |
| 4 | Zuma | Japanese | sushi | 0.9845 |
| 5 | The Breslin | Japanese | sushi | 0.9804 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Sushi Yasuda | Japanese | sushi | 0.9777 |
| 2 | TAO Uptown | Sushi | sushi | 0.95 |
| 3 | Serafina Restaurant | Japanese | sushi | 0.9633 |
| 4 | Jacob's Pickles | Japanese | sushi | 0.962 |
| 5 | Hatsuhana Sushi Restaurant | Japanese | sashimi | 0.9583 |

## japanese_date_night - Find a casual but memorable place for tonight.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 0.98 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 0.98 |
| 3 | Natsumi | Japanese | sushi | 0.9393 |
| 4 | Zuma | Japanese | sushi | 0.9345 |
| 5 | The Breslin | Japanese | sushi | 0.9304 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 1.03 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 1.03 |
| 3 | Natsumi | Japanese | sushi | 0.9893 |
| 4 | Zuma | Japanese | sushi | 0.9845 |
| 5 | The Breslin | Japanese | sushi | 0.9804 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Sushi Yasuda | Japanese | sushi | 0.9777 |
| 2 | TAO Uptown | Sushi | sushi | 0.95 |
| 3 | Serafina Restaurant | Japanese | sushi | 0.9633 |
| 4 | Jacob's Pickles | Japanese | sushi | 0.962 |
| 5 | Hatsuhana Sushi Restaurant | Japanese | sashimi | 0.9583 |

## japanese_date_night - Recommend something flavorful but not too expensive.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 0.98 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 0.98 |
| 3 | Natsumi | Japanese | sushi | 0.9393 |
| 4 | Zuma | Japanese | sushi | 0.9345 |
| 5 | The Breslin | Japanese | sushi | 0.9304 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Blue Fin | Sushi | sushi | 0.98 |
| 2 | TAO Downtown Restaurant | Japanese | sushi | 0.98 |
| 3 | Hatsuhana Sushi Restaurant | Japanese | sashimi | 0.9583 |
| 4 | Bread & Butter | Japanese | sushi | 0.9415 |
| 5 | Sushi Seki | Japanese | sushi | 0.94 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Hatsuhana Sushi Restaurant | Japanese | sashimi | 0.9583 |
| 2 | TAO Uptown | Sushi | sushi | 0.9 |
| 3 | Bread & Butter | Japanese | sushi | 0.9415 |
| 4 | Sushi Seki | Japanese | sushi | 0.94 |
| 5 | Whole Foods Market | Japanese | sushi | 0.9384 |

## exploratory_low_bubble - I want something good for dinner in NYC.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 0.9731 |
| 2 | Danji | Asian | kimchi | 0.9141 |
| 3 | Sakagura | Japanese | sashimi | 0.9137 |
| 4 | Cafe Habana | Mexican | n/a | 0.9111 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9031 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 1.0231 |
| 2 | Danji | Asian | kimchi | 0.9641 |
| 3 | Sakagura | Japanese | sashimi | 0.9637 |
| 4 | Cafe Habana | Mexican | n/a | 0.9611 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9531 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Kenka | Mediterranean | dumplings | 0.9361 |
| 2 | Allora Fifth Ave | Italian | pizza | 0.9306 |
| 3 | Calexico - Upper East Side | Bar | tacos | 0.8527 |
| 4 | Mad Dog & Beans Mexican Cantina | Mexican | fried rice | 0.9513 |
| 5 | KOBA Korean Bbq | Asian | korean bbq | 0.95 |

## exploratory_low_bubble - Find a casual but memorable place for tonight.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 0.9731 |
| 2 | Danji | Asian | kimchi | 0.9141 |
| 3 | Sakagura | Japanese | sashimi | 0.9137 |
| 4 | Cafe Habana | Mexican | n/a | 0.9111 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9031 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 1.0231 |
| 2 | Danji | Asian | kimchi | 0.9641 |
| 3 | Sakagura | Japanese | sashimi | 0.9637 |
| 4 | Cafe Habana | Mexican | n/a | 0.9611 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9531 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Kenka | Mediterranean | dumplings | 0.9361 |
| 2 | Allora Fifth Ave | Italian | pizza | 0.9306 |
| 3 | Calexico - Upper East Side | Bar | tacos | 0.8527 |
| 4 | Mad Dog & Beans Mexican Cantina | Mexican | fried rice | 0.9513 |
| 5 | KOBA Korean Bbq | Asian | korean bbq | 0.95 |

## exploratory_low_bubble - Recommend something flavorful but not too expensive.

### Simulated accepted-history set
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 0.9731 |
| 2 | Danji | Asian | kimchi | 0.9141 |
| 3 | Sakagura | Japanese | sashimi | 0.9137 |
| 4 | Cafe Habana | Mexican | n/a | 0.9111 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9031 |

### Profile-personalized proxy recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Rosa Mexicano | Mexican | salad | 0.9731 |
| 2 | Danji | Asian | kimchi | 0.9641 |
| 3 | Sakagura | Japanese | sashimi | 0.9137 |
| 4 | Cafe Habana | Mexican | n/a | 0.9111 |
| 5 | El Vez and Burrito Bar | Mexican | tacos | 0.9031 |

### Diversity-constrained recommendations
| Rank | Restaurant | Category | Popular food | Analysis score |
|---:|---|---|---|---:|
| 1 | Kenka | Mediterranean | dumplings | 0.8861 |
| 2 | Allora Fifth Ave | Italian | pizza | 0.8806 |
| 3 | Calexico - Upper East Side | Bar | tacos | 0.8527 |
| 4 | Mad Dog & Beans Mexican Cantina | Mexican | fried rice | 0.9013 |
| 5 | KOBA Korean Bbq | Asian | korean bbq | 0.9 |
