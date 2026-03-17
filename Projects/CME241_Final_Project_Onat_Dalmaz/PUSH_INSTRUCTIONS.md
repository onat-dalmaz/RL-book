# Push instructions (local agent)

Use these steps to add this project to your RL-book repo under `Projects/`.

1. **Unzip the archive** (if you received a .zip or .tar.gz):
   ```bash
   unzip CME241_Final_Project_Onat_Dalmaz.zip
   # or
   tar -xzf CME241_Final_Project_Onat_Dalmaz.tar.gz
   ```

2. **Clone or open your local RL-book repo** and go to its root:
   ```bash
   git clone https://github.com/onat-dalmaz/RL-book.git
   cd RL-book
   ```

3. **Copy the project folder into Projects/**:
   ```bash
   cp -r /path/to/CME241_Final_Project_Onat_Dalmaz Projects/
   ```
   (Replace `/path/to/` with the actual path where you extracted the archive.)

4. **Review the files** (optional):
   ```bash
   ls -la Projects/CME241_Final_Project_Onat_Dalmaz/
   ```

5. **Stage, commit, and push**:
   ```bash
   git add Projects/CME241_Final_Project_Onat_Dalmaz/
   git status
   git commit -m "Add CME241 final project package"
   git push
   ```

Target branch (if applicable): `master`. Remote: `https://github.com/onat-dalmaz/RL-book.git`.
