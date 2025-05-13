# <img src="https://i.imgur.com/e9TqKPz.png" width=25  alt="VALScoreboardTracker Banner"/> VALScoreboardTracker 

<p align="center">
  <img src="https://i.imgur.com/e9TqKPz.png" width=200 alt="VALScoreboardTracker Banner"/>
</p>

<p align="center">
  <img src="https://i.imgur.com/lgDp0TA.jpeg" width=800 alt="VALScoreboardTracker Banner"/>
</p>

<p align="center">
  Extract Valorant end-game scoreboard data from 16:9 english screenshots and convert it into a CSV, optimized for team environments. 
</p>

### 📥 Download Page
Download the latest version from our [GitHub Releases](https://github.com/Felox210/VALScoreboardTracker/releases/tag/stable) page.

## 🚀 How to Use

1. Fill `config.ini` file with your `TEAMTAG` or `PLAYERNAMES`*.
2. Place your desired screenshots** in the `/screenshots` folder.
3. Run `VALScoreboardTracker.exe`.
4. All the stats from the screenshots are now copied to your clipboard and saved in `scoreboard.csv`!
5. Fill your Scrim Tracker with the data and enjoy! 🤙🏼

*\*if teamsorting is set to true, it will look for the team tag in the screenshots. if it is set to false, it will look for the exact player names matches*
*\**screenshots need to be in english 16:9 resolution*
### Example Screenshot

<p align="center">
  <img src="https://i.imgur.com/Cp6F15I.png" width=750 alt="VALScoreboardTracker Example Screenshot"/>
</p>


### Example Output

<p align="center">
  <img src="https://i.imgur.com/0FKyutH.png" width=1000 alt="VALScoreboardTracker Example Table"/>
</p>

---

## 📊 Example Scrim Tracker Spreadsheet

We've created a Google Spreadsheet template you can use to track your team's stats. Here's how to use it.

### 📝 How to Use

1. Make a copy of the example spreadsheet.
2. Run the script locally with your screenshots to extract data.
3. Paste the extracted data into the "Empty Sheet for copying data" (use ; as the separator to automatically split it into columns).
4. Copy the clean data from the "Empty Sheet for copying data" into the main "Stats" sheet.
5. Replace the placeholder names in the yellow tables with your actual player names.

➡️ [Example Scrim Tracker](https://docs.google.com/spreadsheets/d/1N7p1be3Yw2lM5oGfvo3qUTSEXE2f-QyXYMvSeXKH7sc/edit?gid=626882904#gid=626882904)

<p align="center">
  <img src="https://i.imgur.com/LqoWu6C.png" width=1000 alt="VALScoreboardTracker Example Tracker"/>
</p>
<p align="center">
  <img src="https://i.imgur.com/9H8pRmg.png" width=1000 alt="VALScoreboardTracker Example Tracker"/>
</p>
<p align="center">
  <img src="https://i.imgur.com/uIJxuQW.png" width=1000 alt="VALScoreboardTracker Example Tracker"/>
</p>

---

## 🛠 Built By

Lovingly put together by **[Felox](https://x.com/felox210)** & **[summonN](https://x.com/summonhalfa)**.

<sub> Special thanks to [**Aplox**](https://twitter.com/_Aplox) for publishing the original version and **isaacaudet** for the agent recognition library. </sub>

---

## ❓ FAQ

### 🔄 Is the tool up to date? Is Agent X / Maps Y supported? 
The tool is currently updated to VALORANT `Patch 10.0`. \
The last added map was `Abyss`.\
The last added agent was `Tejo`.

### 📥 How can I download the tool?
Download the latest version from our [GitHub Releases](https://github.com/Felox210/VALScoreboardTracker/releases/tag/stable) page.

### 🗺️ How can I add new maps?
To add a new map, add the map name to `VALORANT_MAPS` in the `config.ini` file.

### 🎭 How can I add new agents?
To add a new agent, add a **50x50 PNG portrait** named `agentname.png` to the `/agent-images` folder.

### 💖 How can I support the project?
Donations are a great way to support our work! This project will always remain **open-source and free-to-use**.

<p align="center">
  <a href="https://www.paypal.com/donate?hosted_button_id=DXG78P2TZUEJL">
    <img src="https://www.paypalobjects.com/en_US/IT/i/btn/btn_donateCC_LG.gif" alt="Donate with PayPal button" />
  </a>
</p>



