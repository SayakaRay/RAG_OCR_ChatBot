from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    query_instruction_for_retrieval="",
    use_fp16=False
)

model.model.to('cpu')

model = SentenceTransformer("BAAI/bge-m3")  # หรือ "all-MiniLM-L6-v2"
embedding = model.encode(["--- Chunk 1 --- Discussion CRVO has two types: - Nonischemic (70%): which is characterized by vision that is better than 20/200, 16% progress to nonperfused; 50% resolve completely without treatment; defined as <10 disk diameter (DD) of capillary nonperfusion. - Ischemic (30%): which is defined as more than 10 DD of nonperfusion; patients are usually older and have worse vision; 60% develop iris NV; up to 33% develop neovascular glaucoma; 10% are combined with branch retinal arterial occlusion (usually cilioretinal artery due to low perfusion pressure of choroidal system) [7]. Central retinal vein occlusion is a disease of the old population (age >50 years old). Major risk factors are hypertension, diabetes, and atherosclerosis. Other risk factors are glaucoma, syphilis, sarcoidosis, vasculitis, increased intraorbital or intraocular pressure, hypophema, hyperviscosity syndromes (multiple myeloma, Waldenstrom's macroglobulinemia, and leukemia), high homocysteine levels, sickle cell, and HIV [8]."])