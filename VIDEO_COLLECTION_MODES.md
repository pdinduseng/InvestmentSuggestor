# 📹 Video Collection Modes

## Overview

The Investment Analysis Agent supports **two video collection modes** to give you flexibility in how you gather videos for analysis.

---

## 🕐 Mode 1: Time Window (Recommended)

**Fetches all videos published within a specific time window**

### Configuration
```yaml
settings:
  collection_mode: time_window
  lookback_hours: 48  # Last 48 hours
```

### How It Works
1. Calculates cutoff time: `current_time - lookback_hours`
2. Uses YouTube API's `publishedAfter` parameter
3. Fetches **ALL videos** published after the cutoff (up to safety limit)
4. Ideal for regular, scheduled runs

### Use Cases
- ✅ **Daily/scheduled analysis** - Run every morning to analyze yesterday's videos
- ✅ **Recent trends** - Focus only on latest content
- ✅ **Event-driven** - Analyze videos around specific events (earnings, conferences)
- ✅ **Automated workflows** - Cron jobs, scheduled tasks

### Example Scenarios

**Scenario 1: Daily Morning Report**
```yaml
collection_mode: time_window
lookback_hours: 24  # Last 24 hours
```
Run at 9am daily → Analyzes all videos from previous day

**Scenario 2: Weekend Recap**
```yaml
collection_mode: time_window
lookback_hours: 72  # Last 3 days (Friday-Sunday)
```
Run Monday morning → Analyzes weekend content

**Scenario 3: Real-time Monitoring**
```yaml
collection_mode: time_window
lookback_hours: 2  # Last 2 hours
```
Run every 2 hours → Near real-time analysis

### Output Example
```
📹 COLLECTING VIDEOS
------------------------------------------------------------
  Mode: Time window (last 48 hours)
  Fetching videos published after: 2025-11-10 12:00 UTC

  Fetching from: InvestAnswers
    ✅ Found 3 videos

  Fetching from: Meet Kevin
    ✅ Found 2 videos

📊 Total videos found: 5
```

---

## 📊 Mode 2: Max Videos

**Fetches the N most recent videos regardless of when they were published**

### Configuration
```yaml
settings:
  collection_mode: max_videos
  videos_per_channel: 5  # Most recent 5 videos
```

### How It Works
1. Sorts videos by date (most recent first)
2. Takes top N videos from each channel
3. **Ignores publication date** - could be hours, days, or weeks old
4. Ideal for one-time analysis or backfilling

### Use Cases
- ✅ **First-time setup** - Analyze recent content to establish baseline
- ✅ **Backfill analysis** - Catch up on missed content
- ✅ **Manual runs** - One-off analysis of recent videos
- ✅ **Channel comparison** - Ensure equal number of videos per channel

### Example Scenarios

**Scenario 1: Initial Setup**
```yaml
collection_mode: max_videos
videos_per_channel: 10  # Last 10 videos per channel
```
First run → Analyzes recent history

**Scenario 2: Quick Check**
```yaml
collection_mode: max_videos
videos_per_channel: 3  # Just the 3 most recent
```
Fast analysis for spot-checking

### Output Example
```
📹 COLLECTING VIDEOS
------------------------------------------------------------
  Mode: Max videos (5 most recent per channel)

  Fetching from: InvestAnswers
    ✅ Found 5 videos

  Fetching from: Meet Kevin
    ✅ Found 5 videos

📊 Total videos found: 10
```

---

## 🔄 Comparison

| Feature | Time Window | Max Videos |
|---------|-------------|------------|
| **Filter by date** | ✅ Yes | ❌ No |
| **Consistent timing** | ✅ Yes | ❌ No |
| **Variable results** | ✅ Yes (0-N videos) | ❌ No (always N) |
| **Best for automation** | ✅ Yes | ⚠️ Depends |
| **Handles slow channels** | ✅ Yes (may get 0) | ✅ Yes (gets N) |
| **Handles active channels** | ✅ Yes (gets all) | ⚠️ May miss some |

---

## 🎯 Recommended Usage

### For Production / Scheduled Runs
```yaml
collection_mode: time_window
lookback_hours: 24  # Daily analysis
```
**Why**: Ensures you analyze each video exactly once, no duplicates

### For Manual Testing
```yaml
collection_mode: max_videos
videos_per_channel: 3  # Small sample for quick tests
```
**Why**: Fast, consistent results for development

### For Historical Analysis
```yaml
collection_mode: max_videos
videos_per_channel: 50  # Larger sample
```
**Why**: Get substantial data regardless of recent activity

---

## ⚙️ Advanced Configuration

### Safety Limits
Both modes respect the `max_videos_per_channel` safety limit:

```yaml
settings:
  max_videos_per_channel: 50  # Absolute maximum

  # Time window mode
  lookback_hours: 168  # 1 week
  # If a channel posted 100 videos this week, only gets first 50

  # Max videos mode
  videos_per_channel: 10
  # Tries to get 10, but never exceeds 50
```

### Combining with Other Settings

```yaml
settings:
  # Video collection
  collection_mode: time_window
  lookback_hours: 48
  max_videos_per_channel: 50

  # Analysis filters
  min_confidence: 0.7  # Only high-confidence stocks
  max_cost_per_run: 5.00  # Cost limit

  # LLM settings
  analysis_mode: gemini  # Use Gemini for video analysis
```

---

## 📅 Practical Examples

### Example 1: Daily Investment Newsletter

**Goal**: Send daily email with stock recommendations from yesterday

```yaml
settings:
  collection_mode: time_window
  lookback_hours: 24
  min_confidence: 0.8
```

**Cron schedule**: `0 6 * * *` (6am daily)

**Result**: Analyzes all videos from past 24 hours, high-confidence stocks only

---

### Example 2: Weekend Analysis

**Goal**: Analyze all weekend content on Monday

```yaml
settings:
  collection_mode: time_window
  lookback_hours: 72  # Friday 5pm to Monday 5pm
```

**Schedule**: Monday 6pm

**Result**: Catches all weekend uploads

---

### Example 3: New Channel Evaluation

**Goal**: Analyze a new channel's recent content

```yaml
channels:
  - channel_id: "NEW_CHANNEL_ID"
    name: "New Channel"

settings:
  collection_mode: max_videos
  videos_per_channel: 20  # Last 20 videos
```

**Result**: Get overview of channel's typical content

---

## 🐛 Troubleshooting

### "Found 0 videos" in Time Window Mode

**Cause**: No videos published in the lookback window

**Solutions**:
1. Increase `lookback_hours`
2. Switch to `max_videos` mode temporarily
3. Check if channels are active

### Too Many Videos

**Issue**: Time window returns 100+ videos

**Solutions**:
1. Reduce `lookback_hours`
2. Adjust `max_videos_per_channel` safety limit
3. Set `max_cost_per_run` to stop analysis early

### Missing Recent Videos

**Issue**: Using `max_videos` mode but new videos are missing

**Solution**: Switch to `time_window` mode - it's more reliable for recent content

---

## 🔍 YouTube API Quota Impact

Both modes use YouTube API quota:

**Time Window Mode**:
- 1 search request per channel = 100 quota units
- Example: 3 channels = 300 units

**Max Videos Mode**:
- Same: 100 units per channel

**Daily Quota**: 10,000 units (free tier)
- Can analyze ~30 channels per day (100 units × 30 = 3000 units)
- Or run 30+ times with 3 channels (100 units × 3 × 30 = 9000 units)

---

## 💡 Best Practices

1. **Use `time_window` for automation** - Prevents duplicate analysis
2. **Set reasonable safety limits** - Protect against API quota exhaustion
3. **Start with shorter windows** - Test with 6-12 hours before going to 48+
4. **Monitor results** - Check how many videos each mode returns
5. **Combine with cost limits** - Protect against unexpected LLM costs

---

## 📈 Future Enhancements

Potential additions:
- [ ] **Date range mode**: Specific start/end dates
- [ ] **Incremental mode**: Track last run, only fetch new videos
- [ ] **Channel-specific settings**: Different modes per channel
- [ ] **Playlist support**: Analyze specific playlists
- [ ] **Live stream detection**: Filter out/include live streams

---

**Default**: Time window mode with 48-hour lookback - ideal for most use cases!
