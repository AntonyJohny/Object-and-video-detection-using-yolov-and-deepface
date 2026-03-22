from deepface import DeepFace
import os
import pickle

# --- Configuration ---
DATASET_PATH = "face_dataset"
DATABASE_FILE = "face_database.pkl"
MODEL_NAME = "VGG-Face"

def enroll_faces():
    """
    Analyzes all images in the dataset path, generates face embeddings,
    and saves them to a pickle file for fast lookup.
    """
    face_database = {}

    # 1. Check if the dataset folder exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset folder not found at '{DATASET_PATH}'")
        print("Please create the folder and add subfolders with names and images.")
        # Create the folder structure for the user if it doesn't exist
        os.makedirs(DATASET_PATH, exist_ok=True)
        print(f"Created empty folder: {DATASET_PATH}. Please add photos inside.")
        return

    # 2. Loop through each person's folder
    for person_name in os.listdir(DATASET_PATH):
        person_folder_path = os.path.join(DATASET_PATH, person_name)
        
        # Skip files, only process directories
        if not os.path.isdir(person_folder_path):
            continue

        print(f"Processing images for: {person_name}")
        person_embeddings = []
        
        # 3. Loop through images in the person's folder
        for image_name in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_name)
            
            # Basic check for image extensions
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # Generate the embedding for the face in the image
                # enforce_detection=True ensures we skip images where no face is found
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True
                )
                # DeepFace returns a list (one per face), we take the first one
                person_embeddings.append(embedding_objs[0]["embedding"])
                print(f"  - ✅ Successfully processed {image_name}")
            except Exception as e:
                print(f"  - ❌ Could not process {image_name}. Reason: {e}")
        
        # Only add the person if we successfully got at least one embedding
        if person_embeddings:
            face_database[person_name] = person_embeddings

    if not face_database:
        print("\n❌ No faces were enrolled. Please check that your 'face_dataset' folder has subfolders with valid images.")
        return

    # 4. Save the database to disk
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(face_database, f)
    
    print(f"\n✅ Enrollment complete. Database saved to '{DATABASE_FILE}' with {len(face_database)} person(s).")

if __name__ == "__main__":
    enroll_faces()