from django import forms

class VideoUploadForm(forms.Form):

    upload_video_file = forms.FileField(label="Select Video", required=True,widget=forms.FileInput(attrs={"accept": "video/*"}))
    sequence_length = forms.IntegerField(label="Sequence Length", required=True)
from django import forms

class ImageUploadForm(forms.Form):
    upload_image_file = forms.ImageField(
        label='Select Image',
        required=True,
        widget=forms.FileInput(attrs={
            "accept": "image/*",
            "class": "custom-file-input",  # Add custom class for styling
            "id": "image-upload-input"      # Add an id for JavaScript use
        })
    )
   

# class TextUploadForm(forms.Form):
#     upload_text_file = forms.FileField(
#         label="Select Text File", 
#         required=True,
#         widget=forms.FileInput(attrs={"accept": ".txt,.docx,.pdf"})
#     )
from django import forms

class TextUploadForm(forms.Form):
    upload_text_file = forms.FileField(
        label="Select Text File", 
        required=True,
        widget=forms.FileInput(attrs={"accept": ".txt,.docx,.pdf"})
    )


