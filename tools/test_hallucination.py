from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

def detect_hallucinations(response, context):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    response_vector = embeddings.embed_query(response)
    context_vector = embeddings.embed_query(context)
    similarity_score = cosine_similarity([response_vector], [context_vector])[0][0]
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    common_words = response_words.intersection(context_words)
    is_hallucination = similarity_score < 0.1 or len(common_words) == 0
    
    result = {
        "is_hallucination": is_hallucination,
        "similarity_score": similarity_score,
        "common_words": list(common_words)
    }
    
    return result

# Sử dụng hàm
response = "รัดย่องมีการที่สมินวัด"
context = 'A. Phí khởi tạo dịch vụ và đào tạo\n\n\n\n\n\n\n\n\n\n\n\n1\nPhí khởi tạo hệ thống\nDịch vụ\n2000000\n1\n2000000\n\n0\n2000000\nPhí khởi tạo dịch vụ bao gồm: phí cấu hình hệ thống và phí khởi tạo dữ liệu ban đầu Thanh toán một lần duy nhất Buổi\n1000000\n3\n3000000\n0.666667\n2000000\n1000000\nChi phí đào tạo triển khai tính theo số lớp khách hàng đăng ký đào tạo: 1.000.000 đồng / 01 buổi áp dụng cho đào tạo trực tiếp khu vực Hà Nội và Thành phố Hồ Chí Minh và trực tuyến ở tất cả các tỉnh/thành phố / 1 năm\\n- Dịch vụ hỗ trợ trực tuyến trong suốt thời gian vận hành miễn phí\\n- Thời gian đăng ký sử dụng tối thiểu 12 tháng. Thanh toán phí sử dụng 1 lần vào đầu kỳ thanh toán. 15\n\nCảnh báo khối lượng thực tế vượt khối lượng kế hoạch bằng text màu đỏ cả dòng công việc\n\n\n16\nHợp đồng giao thầu\nQuản lý chi tiết thông tin hợp đồng ký với thầu phụ: GTHĐ,phụ lục đính kèm, và bảng khối lượng thi công từng hạng mục(BOQ) ký với thầu phụ'

result = detect_hallucinations(response, context)
print(result)