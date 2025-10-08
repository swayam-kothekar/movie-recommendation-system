# ─────────────────────────────────────────────
# MUSIC RECOMMENDATION SYSTEM USING SPARK + R
# ─────────────────────────────────────────────

# 1 LOAD LIBRARIES
suppressPackageStartupMessages({
  library(sparklyr)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
})

# 2 CONNECT TO SPARK
sc <- spark_connect(master = "local", app_name = "MusicRecommendationSystem")

cat("\n", strrep("=", 60), "\n")
cat("MUSIC RECOMMENDATION SYSTEM USING SPARK + R\n")
cat(strrep("=", 60), "\n\n")

# 3 LOAD DATA
# Change to your local CSV path
data_path <- "/home/swayam/Downloads/archive/Last.fm_data.csv"

df <- spark_read_csv(sc, name = "music_data", path = data_path, header = TRUE, infer_schema = TRUE)
record_count <- df %>% tally() %>% collect()
cat(paste0("Data loaded successfully (", record_count$n, " records)\n\n"))

# 4 BASIC STATS
user_count <- df %>% summarise(users = n_distinct(Username)) %>% collect()
track_count <- df %>% summarise(tracks = n_distinct(Track)) %>% collect()
cat(paste0("Unique Users: ", user_count$users, "\n"))
cat(paste0("Unique Tracks: ", track_count$tracks, "\n"))

# 5 PREPROCESSING — AGGREGATE PLAY COUNTS
agg_data <- df %>%
  group_by(Username, Track) %>%
  summarise(rating = n()) %>%
  rename(user_id = Username, track_id = Track)

agg_data_tbl <- sdf_register(agg_data, "ratings_tbl")
cat(paste0("Aggregated play counts — total user-track pairs: ", sdf_nrow(agg_data_tbl), "\n"))

# 6 CONVERT USER AND TRACK TO NUMERIC (INDEX ENCODING)
cat("\nEncoding user_id and track_id to numeric indices...\n")

agg_indexed <- agg_data_tbl %>%
  ft_string_indexer(input_col = "user_id", output_col = "user_index") %>%
  ft_string_indexer(input_col = "track_id", output_col = "track_index")

# 7 TRAIN ALS MODEL
cat("\nTraining ALS collaborative filtering model...\n")

als_model <- ml_als(
  agg_indexed,
  rating_col = "rating",
  user_col = "user_index",
  item_col = "track_index",
  rank = 10,
  max_iter = 10,
  reg_param = 0.1,
  implicit_prefs = TRUE
)

cat("Model trained successfully!\n")

# 8 GENERATE TOP-5 RECOMMENDATIONS (NEW SPARKLYR API)
cat("\nGenerating recommendations...\n")

user_recs  <- ml_recommend(als_model, type = "users")
track_recs <- ml_recommend(als_model, type = "items")

cat("Generated Top-5 recommendations for users and tracks\n")

# 9 CONVERT TO R DATAFRAMES
user_recs_df <- user_recs %>% collect()
agg_df <- agg_data %>% collect()

# 10 CREATE OUTPUT FOLDER
dir.create("output_r", showWarnings = FALSE)

# 11 VISUAL ANALYSIS
cat("\nGenerating and saving visual analysis...\n")

# 11.1 Rating Distribution
p1 <- ggplot(agg_df, aes(x = rating)) +
  geom_histogram(bins = 15, fill = "skyblue", color = "black") +
  ggtitle("Distribution of Play Counts per Song") +
  xlab("Play Count (Implicit Rating)") +
  ylab("Frequency")
ggsave("output_r/rating_distribution.png", plot = p1, width = 8, height = 4)

# 11.2 Top 10 Tracks
top_tracks <- agg_df %>%
  group_by(track_id) %>%
  summarise(total_plays = sum(rating)) %>%
  arrange(desc(total_plays)) %>%
  head(10)

p2 <- ggplot(top_tracks, aes(x = reorder(track_id, total_plays), y = total_plays)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  ggtitle("Top 10 Most Listened Tracks") +
  xlab("Track Name") + ylab("Total Play Count")
ggsave("output_r/top_10_tracks.png", plot = p2, width = 8, height = 5)

# 11.3 Top 10 Users
top_users <- agg_df %>%
  group_by(user_id) %>%
  summarise(total_plays = sum(rating)) %>%
  arrange(desc(total_plays)) %>%
  head(10)

p3 <- ggplot(top_users, aes(x = reorder(user_id, total_plays), y = total_plays)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  ggtitle("Top 10 Most Active Users") +
  xlab("User ID") + ylab("Total Play Count")
ggsave("output_r/top_10_users.png", plot = p3, width = 8, height = 5)

# 11.4 Recommendations Heatmap
cat("Flattening recommendations for visualization...\n")
rec_flat <- user_recs_df %>% tidyr::unnest(recommendations)

p4 <- ggplot(rec_flat, aes(x = as.factor(user_index), y = as.factor(track_index), fill = rating)) +
  geom_tile() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  ggtitle("User vs Track Recommendation Strength") +
  xlab("User Index") + ylab("Track Index") +
  theme(axis.text.x = element_blank(), axis.text.y = element_blank())
ggsave("output_r/recommendation_heatmap.png", plot = p4, width = 7, height = 5)

cat("Visualization files saved in 'output_r/'\n")

# 12 SAVE OUTPUT FILES
write_csv(top_tracks, "output_r/top_10_tracks.csv")
write_csv(top_users, "output_r/top_10_users.csv")
write_csv(rec_flat, "output_r/user_recommendations.csv")

cat("Data and charts saved locally in /output_r\n")

# 13 SUMMARY PRINT
cat("\n", strrep("=", 60), "\n")
cat("MUSIC RECOMMENDATION SYSTEM EXECUTED SUCCESSFULLY\n")
cat(strrep("=", 60), "\n")
cat("Generated files:\n")
cat(" - rating_distribution.png\n")
cat(" - top_10_tracks.png\n")
cat(" - top_10_users.png\n")
cat(" - recommendation_heatmap.png\n")
cat(" - top_10_tracks.csv\n")
cat(" - top_10_users.csv\n")
cat(" - user_recommendations.csv\n")

# 14 DISCONNECT FROM SPARK
spark_disconnect(sc)

